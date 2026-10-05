# Calibrates the two model.py constants that decide most picks, on a
# for_each_tile loop (v3's 0.688 was measured on hint-driven coarse tiling).
# One-off calibration: it compiles once per EG, which is fine here because it
# is not part of the cost model's budget.
#
# Sweep A -> PER_ITER_OVERHEAD_US
#   x + y over (CG, 256), no weights, far below LX. Only the trip count
#   changes with EG, so   time = a * trips + b   and  a  is the per-iteration
#   overhead.  Same op/shape family as calibrate_loop_overhead_v3.py.
#
# Sweep B -> WEIGHT_READ_GBS
#   x @ W.T with W (4096, 4096) fp16 = 32 MB invariant, x (CG, 4096). Each
#   trip re-reads W, so   slope_B = a + W_bytes / BW   and
#   BW = W_bytes / (slope_B - a).  Tells you whether weights come in at the
#   150 GB/s read peak or at the replicated per-core rate (~2.3 GB/s/core)
#   that cost_model.py documents for M-split matmul operands.
#
# Not calibrated here: SPILL_GBS / SPILL_PASSES (needs a case where only the
# spill changes with EG, which is hard to build cleanly -- see README).
#
# Run on the Spyre VM:  python3 costmodel2.0/calibrate_for_each_tile.py

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

import capture  # noqa: E402
from workloads import Workload  # noqa: E402

CG = 3840  # divisible by 1,2,3,4,5,6,8,10,12,15,16,20,24,30,32,...
TRIPS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]


def _fit(points):
    n = len(points)
    mx = sum(x for x, _ in points) / n
    my = sum(y for _, y in points) / n
    sxx = sum((x - mx) ** 2 for x, _ in points)
    a = sum((x - mx) * (y - my) for x, y in points) / sxx
    b = my - a * mx
    ss_res = sum((y - (a * x + b)) ** 2 for x, y in points)
    ss_tot = sum((y - my) ** 2 for _, y in points)
    return a, b, (1 - ss_res / ss_tot) if ss_tot else float("nan")


def sweep(w: Workload, cg: int, trips_list):
    args = tuple(t.to(capture.DEVICE_NAME) for t in w.make_operands(cg))
    points = []
    for trips in trips_list:
        eg = cg // trips
        try:
            fn, info = capture.compile_and_capture(w, eg, args)
        except Exception as e:  # noqa: BLE001
            print(f"  trips={trips:<3} EG={eg:<5} COMPILE FAILED: {str(e).splitlines()[0][:200]}")
            continue
        us = capture.measure(fn, args, f"trips={trips:<3} EG={eg:<5}")
        if us is not None:
            points.append((trips, us))
    if len(points) < 3:
        print("  too few points to fit")
        return None
    a, b, r2 = _fit(points)
    print(f"  fit: time = {a:.3f} * trips + {b:.1f} us   (r^2 = {r2:.3f})")
    return a


def main():
    ADD = Workload(
        name="cal_add", description="", cg=CG,
        make_operands=lambda n: ((torch.randn(n, 256)).half(), (torch.randn(n, 256)).half()),
        dims=(0, 0), body=lambda x, y: x + y,
    )
    D = 4096
    LINEAR = Workload(
        name="cal_linear", description="", cg=1024,
        make_operands=lambda n: ((torch.randn(n, D) * 0.1).half(),
                                 (torch.randn(D, D) * D**-0.5).half()),
        dims=(0, None), body=lambda x, w: x @ w.T,
    )

    print(f"=== Sweep A: x + y, ({CG}, 256), trips {TRIPS}")
    a = sweep(ADD, CG, TRIPS)
    if a is not None:
        print(f"  -> PER_ITER_OVERHEAD_US = {a:.3f}   (model.py currently 0.688)")

    trips_b = [1, 2, 4, 8, 16]
    print(f"\n=== Sweep B: x @ W.T, x ({LINEAR.cg}, {D}), W ({D}, {D}) invariant, trips {trips_b}")
    slope_b = sweep(LINEAR, LINEAR.cg, trips_b)
    if a is not None and slope_b is not None and slope_b > a:
        w_bytes = D * D * 2
        bw = w_bytes / ((slope_b - a) * 1e3)
        print(f"  -> WEIGHT_READ_GBS = {bw:.1f}   (model.py currently 150; "
              f"replicated per-core rate would be ~{2.3 * 32:.0f})")


if __name__ == "__main__":
    main()
