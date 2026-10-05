# mark_dynamic + for_each_tile -> ConstraintViolationError
#
# mark_dynamic(min=128, max=512) tells PyTorch that one compiled binary must
# serve every integer from 128 to 512. for_each_tile then checks that the size
# divides evenly into tiles, and since the size is a symbol at that point,
# Dynamo records S % 64 == 0 as a guard.
#
# 129 is in [128, 512] and is not a multiple of 64, so the guard and the
# constraint contradict each other and nothing compiles -- not even at a valid
# size like 256.
#
# Run: python repro_constraint_violation.py
import torch
import torch._dynamo

import torch_spyre  # noqa: F401
from torch_spyre._inductor.wsr import for_each_tile

G = 64


def tiled_abs(x):
    _, out = for_each_tile(
        lambda _carry, tiles: (None, torch.abs(tiles[0])),
        (x,),
        dims=(0,),      # slice x along dim 0
        tile_size=G,
        out_dim=0,      # map mode
    )
    return out


compiled = torch.compile(tiled_abs, fullgraph=True)

x = torch.randn(256, 128, dtype=torch.float16, device="spyre")
torch._dynamo.mark_dynamic(x, 0, min=128, max=512)

try:
    compiled(x)
    print("compiled")
except Exception as e:
    print(f"{type(e).__name__}")
    for line in str(e).splitlines():
        if line.strip().startswith("-"):
            print(f"  {line.strip()}")
