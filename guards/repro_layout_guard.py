# The Spyre tensor layout guard recompiles on every new size.
#
# All four sizes are inside [64, 512], so Dynamo's range guard passes. But the
# size sits inside the Spyre layout (device_size=[2, S, 64]), and torch-spyre's
# layout guard requires an exact match -- so each size recompiles anyway.
#
# Run: TORCH_LOGS="recompiles" python repro_layout_guard.py
import torch
import torch._dynamo

import torch_spyre  # noqa: F401


@torch.compile
def f(x):
    return x * 2


for s in [128, 256, 320, 448]:
    x = torch.randn(s, 128, dtype=torch.float16, device="spyre")
    torch._dynamo.mark_dynamic(x, 0, min=64, max=512)
    print(f"S={s}  {x.device_tensor_layout()}")
    try:
        f(x)
    except RuntimeError as e:
        # #2434: the runtime does not yet pass the extra size argument that a
        # dynamic graph expects. Unrelated to guards -- the compile already
        # happened by this point, which is what we are measuring.
        print(f"  runtime: {str(e).splitlines()[0][:70]}")
