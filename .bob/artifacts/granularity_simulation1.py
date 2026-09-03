import math
import sympy

LX_PER_CORE = 1_625_344  # bytes; = 2MB - 64KB reserved, minus DeepTools' 20% share, rounded to stick

ELEMS_PER_STICK_FP16 = 64   # 128-byte stick / 2 bytes per fp16 element
ITEMSIZE_FP16        = 2    # bytes per fp16 element


def choose_granularity_v1(CG: int, n_cores: int, hidden: int) -> int:
    """Model 1: largest divisor of CG whose per-core buffer fits in LX.
    """
    for EG in sorted(sympy.divisors(CG), reverse=True):
        rows_per_core = EG // n_cores
        padded_hidden = math.ceil(hidden / ELEMS_PER_STICK_FP16) * ELEMS_PER_STICK_FP16
        tile_bytes    = rows_per_core * padded_hidden * ITEMSIZE_FP16

        if tile_bytes <= LX_PER_CORE:
            return EG

    return 1  # smallest possible divisor, last resort

# ── Try it ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"LX_PER_CORE = {LX_PER_CORE:,} bytes\n")
    print(choose_granularity_v1(CG=576, n_cores=32, hidden=32000))
    print(choose_granularity_v1(CG=2048, n_cores=32, hidden=32000))