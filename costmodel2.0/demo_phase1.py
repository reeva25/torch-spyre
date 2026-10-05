# Phase 1 demo for cost model 2.0 -- same shape as costmodel/demo_*.py.
#
# The outermost dimension has concrete size CG (tile size == granularity in the
# symbolic-shapes design). Vary it with --cg.
#
#   1. Baseline: for_each_tile(tile_size=CG) -- one trip. Capture the
#      residency-filtered buffer list (+ which buffers hold rows of the tiled
#      axis) BEFORE LX planning. Measure device time.
#   2. Cost model: score every EG dividing CG from that ONE capture
#      (model.py -- arithmetic only, no compile, no solver).
#   3. Test only: recompile ONCE at the picked EG, measure, check the result
#      matches, and compare the model's PREDICTED peak bytes at that EG with
#      the peak ACTUALLY captured from the recompile.
#
# Run on the Spyre VM (from anywhere inside the repo checkout):
#   python3 costmodel2.0/demo_phase1.py --workload mlp
#   python3 costmodel2.0/demo_phase1.py --workload mlp --cg 1024
#   python3 costmodel2.0/demo_phase1.py --workload mlp --also-eg 128,512
#       (--also-eg: extra EGs to recompile and grade; one compile each)

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

import model  # noqa: E402
import workloads  # noqa: E402

DEFAULT_MIN_EG = 8  # under 8 rows a tile under-fills a core's 8-row pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", required=True, choices=list(workloads.ALL))
    ap.add_argument("--cg", type=int, help="outermost size == granularity (default: workload's)")
    ap.add_argument("--min-eg", type=int, default=DEFAULT_MIN_EG)
    ap.add_argument("--force-eg", type=int, help="skip the model's pick and test this EG")
    ap.add_argument("--also-eg", default="", help="extra EGs to recompile and grade")
    ap.add_argument("--allow-all-ops-in-lx", action="store_true",
                    help="the LX test suites set this; production default is off")
    args = ap.parse_args()

    import capture  # torch_spyre import deferred so --help works anywhere

    w = workloads.ALL[args.workload]
    cg = args.cg or w.cg
    params = model.Params(lx_capacity_bytes=capture.lx_capacity_bytes())

    print(f"=== {w.name}: {w.description}")
    print(f"    outermost size = CG = {cg};  LX = {params.lx_capacity_bytes:,} B/core\n")

    torch.manual_seed(0)
    args_dev = tuple(t.to(capture.DEVICE_NAME) for t in w.make_operands(cg))

    # ── 1. baseline: tile = CG ──────────────────────────────────────────────
    print(f"=== 1. baseline compile, tile_size = CG = {cg} (1 trip)")
    base_fn, base_info = capture.compile_and_capture(w, cg, args_dev, args.allow_all_ops_in_lx)
    print(f"  compile {base_info['compile_s']:.1f}s")
    capture.print_buffers(base_info, params.lx_capacity_bytes)
    base_us = capture.measure(base_fn, args_dev, f"EG={cg} (baseline)")
    base_out = base_fn(*args_dev).cpu()
    cap = capture.to_model_capture(w, cg, base_info, args_dev)

    # ── 2. cost model, from that one capture ────────────────────────────────
    print(f"\n=== 2. cost model (one capture, no compiles)  "
          f"invariant bytes = {cap.invariant_bytes:,}")
    model.print_table(cap, args.min_eg, params)
    eg_formula = model.pick(cap, args.min_eg, params)
    eg_fit = model.pick_largest_fitting(cap, args.min_eg, params)
    print(f"\n  formula picks EG={eg_formula};  largest-fitting rule picks EG={eg_fit}")

    # ── 3. test: recompile at the pick (and any extras), compare ───────────
    chosen = args.force_eg or eg_formula
    to_test = []
    for eg in [chosen] + [int(x) for x in args.also_eg.split(",") if x]:
        if cg % eg:
            print(f"\n  skipping EG={eg}: does not divide CG={cg}")
        elif eg != cg and eg not in to_test:
            to_test.append(eg)

    results = {cg: base_us}
    if not to_test:
        print(f"\n=== 3. model keeps EG = CG = {cg}; nothing to recompile.")
    for eg in to_test:
        print(f"\n=== 3. recompile at EG={eg}, trips = {cg // eg}")
        try:
            fn, info = capture.compile_and_capture(w, eg, args_dev, args.allow_all_ops_in_lx)
        except Exception as e:  # noqa: BLE001 -- keep going to the summary
            print(f"  COMPILE FAILED: {type(e).__name__}: {str(e).splitlines()[0][:300]}")
            results[eg] = None
            continue
        print(f"  compile {info['compile_s']:.1f}s")
        capture.print_buffers(info, params.lx_capacity_bytes)
        predicted = model.peak_live_bytes(model.buffers_at(cap, eg))
        actual = model.peak_live_bytes(info["buffers"])
        err = (predicted - actual) / actual if actual else float("nan")
        print(f"  peak bytes/core: predicted {predicted:,} from the CG capture, "
              f"actual {actual:,} from this compile  ({err:+.0%})")
        results[eg] = capture.measure(fn, args_dev, f"EG={eg}")
        try:
            torch.testing.assert_close(fn(*args_dev).cpu(), base_out, atol=w.atol, rtol=w.rtol)
            print("  result matches the baseline")
        except AssertionError as e:
            print(f"  !! RESULT DIFFERS from the baseline: {str(e).splitlines()[0]}")

    # ── summary ──────────────────────────────────────────────────────────────
    # "predicted" is the EG-dependent part only (loop + weights + spill); the
    # row streaming that every EG pays equally is not in it, so compare the
    # predicted DIFFERENCES between rows with the measured differences.
    print("\n=== summary")
    for eg, us in results.items():
        pred = model.predict(cap, eg, params).total_us
        tag = " (baseline)" if eg == cg else (" (model pick)" if eg == chosen else "")
        gain = f"{(base_us - us) / base_us:+.1%} vs baseline" if us and base_us and eg != cg else ""
        meas = f"{us:,.1f}" if us else "-"
        print(f"  EG={eg:<5}{tag:<14} predicted {pred:>10,.1f} us   measured {meas:>10} us   {gain}")


if __name__ == "__main__":
    main()
