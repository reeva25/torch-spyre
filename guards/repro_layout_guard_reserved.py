# Does the max reservation make the Spyre layout guard size-independent?
#
# repro_layout_guard.py showed four sizes giving four compiles, with only the
# layout guard named in the recompile reason. The layout carries the row count
# (device_size=[2, S, 64]), so the guard's exact-match comparison fails at every
# new size.
#
# Review claim: with the max reservation (#4326) the layout is built from the
# padded max shape, so device_size is [2, 512, 64] at every runtime size, the
# unmodified guard passes, and one binary is reused.
#
# This script runs the same sizes twice, once per spelling, and counts compiles
# directly instead of reading them out of the logs.
#
# Run: python repro_layout_guard_reserved.py
# Optionally with TORCH_LOGS="recompiles" for the guard-failure reasons.
import torch
import torch._dynamo

import torch_spyre  # noqa: F401

SIZES = [128, 256, 320, 448, 512]
G = 64
MAX = 512


def make_counting_backend(counter):
    def backend(gm, example_inputs):
        counter.append(gm)
        return gm.forward

    return backend


def run(label, to_kwargs):
    """Compile f over every size in SIZES, return the number of compiles."""
    print(f"\n=== {label} ===")
    torch._dynamo.reset()
    compiles = []

    def f(x):
        # The supported spelling: the range and the granularity are declared
        # inside the trace as guards, so no StrictMinMaxConstraint exists for
        # the divisibility guard to contradict.
        n = x.size(0)
        torch._check(n >= 2 * G)
        torch._check(n <= MAX)
        torch._check(n % G == 0)
        return x * 2

    compiled = torch.compile(f, backend=make_counting_backend(compiles))

    for s in SIZES:
        host = torch.randn(s, 128, dtype=torch.float16)
        x = host.to("spyre", **to_kwargs)
        torch._dynamo.mark_dynamic(x, 0)  # bare: no min=, no max=
        before = len(compiles)
        layout = x.device_tensor_layout()
        try:
            compiled(x)
        except RuntimeError as e:
            # #2434: the runtime does not yet pass the extra size argument a
            # dynamic graph expects. The compile has already happened by this
            # point, which is what we are measuring.
            print(f"  S={s:4d}  runtime: {str(e).splitlines()[0][:60]}")
        compiled_now = len(compiles) > before
        print(f"  S={s:4d}  {layout}  compiled={compiled_now}")

    print(f"  -> {len(compiles)} compile(s) for {len(SIZES)} sizes")
    return len(compiles)


plain = run('plain .to("spyre")', {})

try:
    reserved = run(f'.to("spyre", max={MAX})', {"max": MAX})
except TypeError as e:
    print(f"\n=== .to(\"spyre\", max={MAX}) ===")
    print(f"  not available in this build: {e}")
    print("  (#4326 reservation. Re-run on an image that has it.)")
    reserved = None

print("\n=== verdict ===")
print(f"  plain:    {plain} compile(s)")
if reserved is None:
    print("  reserved: not available in this build")
else:
    print(f"  reserved: {reserved} compile(s)")
    if reserved == 1:
        print("  -> the unmodified layout guard is already size-independent.")
        print("     Guard 1 is not needed for the reserved dim 0 case.")
    else:
        print("  -> still recompiling. Run with TORCH_LOGS=recompiles to see")
        print("     which guard is named, and report that.")
