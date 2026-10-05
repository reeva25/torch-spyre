# Symbolic Shapes: Guards Design


## Bug

Before the guard work: there is an error thrown when `for_each_tile` and
`mark_dynamic` are run together.

When you call `mark_dynamic`, PyTorch records a **constraint** saying that every
value from min to max must be served by one compiled binary. `for_each_tile`
then checks that the size divides evenly into tiles, and because the size is a
symbol at that point, Dynamo records `S % 64 == 0` as a guard.

Not every value `mark_dynamic` promised is divisible by 64 - 129 is in
`[128, 512]` and isn't. That conflict raises `ConstraintViolationError`, and the
compilation never happens.

### Reproducer

`guards/repro_constraint_violation.py`:

```python
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

compiled(x)
```

Note the call is at S=256, which is a perfectly valid size, 4 whole tiles. It
still fails, because the constraint is checked against the whole declared range,
not against the size actually passed.

Output:

```text
ConstraintViolationError
  - Not all values of L['x'].size()[0] in the specified range
    128 <= L['x'].size()[0] <= 512 satisfy the generated guard
    (L['x'].size()[0] % 64) == 0.
  - ... satisfy the generated guard L['x'].size()[0] == (64*(L['x'].size()[0] // 64)).
  - ... satisfy the generated guard (L['x'].size()[0] % (L['x'].size()[0] // 64)) == 0.
  - ... satisfy the generated guard ((L['x'].size()[0] // 64) % (64*(L['x'].size()[0] // 64))) != 0.
```

Four guards, all consequences of divisibility. The first comes from
`for_each_tile`'s explicit check
([for_each_tile.py:163](torch_spyre/_inductor/wsr/for_each_tile.py#L163)); the
rest come from the `unflatten` that builds the stack of tiles
([line 230](torch_spyre/_inductor/wsr/for_each_tile.py#L230)), since a reshape
is only legal if the element count is unchanged.

Dropping `min` to 64 adds a fifth, `(S // 64) != 1`, that is guard 2 below.

Two workarounds:

**1. Monkey-patch the constraint.** Patch where `mark_dynamic` records the
constraint so it can carry a granularity, and patch the check that validates
guards against it so divisibility is accepted rather than flagged. (This involves
edits to PyTorch internals)

**2. Mark the loop count instead of the row count.** View `(S, 128)` as
`(S//G, G, 128)` and mark dim 0 of that. The symbol is then the loop count, and
the row count becomes `G*k`, so divisibility is provable and no guard is
generated.

Less invasive, but two things need looking into:

- The flatten back to `(S, 128)` has to happen inside the compiled path, which
  may not be easy.
- How it works with multiple symbolic dimensions (Phase 2).

**Where I need input:** is workaround 2 worth looking into further, or should I
focus on the monkey patch?

## The guards

### 1. Adapt the Spyre tensor layout guard

Not a guard I am designing. This is a fix to a guard that already exists, so
that it stops breaking the dynamic case. It belongs with functional enablement
rather than with guard design, but nothing else currently covers it.

torch-spyre adds a guard checking that a tensor's Spyre layout matches the one
recorded at compile time ([_monkey_patch.py:350](torch_spyre/_monkey_patch.py#L350)).
The size sits inside that layout, so every new size fails the guard and
recompiles, which cancels symbolic shapes.

This needs adapting for the symbolic case so it overlooks the varying dimension.
A change in dtype, element arrangement or storage offset should still recompile,
because then the layout genuinely did change.

#### The fix

The guard is installed by `_spyre_TENSOR_MATCH`, which wraps PyTorch's own
`GuardBuilder.TENSOR_MATCH` and attaches a lambda to every Spyre tensor
([_monkey_patch.py:378](torch_spyre/_monkey_patch.py#L378)):

```python
tensor_guard_manager.add_lambda_guard(
    lambda x: (
        x.device.type != DEVICE_NAME
        or (
            x.device_tensor_layout() == expected_layout
            and x.storage_offset() == expected_offset
        )
    ),
    ...
)
```

The `==` compares the whole layout object, including the entry that holds the
size. The fix is to compare the fields individually and skip that one entry:

| Field | Action |
|---|---|
| `device_size`, the dynamic dim's entry | skip |
| `device_size`, every other entry | compare exactly |
| `stride_map` | compare exactly |
| `device_dtype` | compare exactly |
| `element_arrangement` | compare exactly |
| `storage_offset` | compare exactly |

`stride_map` needs no special handling. It was identical at every size in the
run below, because nothing strides over the outermost dimension.

Which entry to skip has to be worked out, because the layout object discards
its `dim_map` at construction
([coarse_tile.py:1948](torch_spyre/_inductor/wsr/coarse_tile.py#L1948)), so the
host dim to device dim mapping is not carried on it. In the run below the entry
that varies is `device_size[1]`, the row count. Confirming how to recover that
in general is an open item.

Two other places compare the same layout the same way and need the same
treatment:

| Where | What it is | Effect with a dynamic dim |
|---|---|---|
| [_monkey_patch.py:431](torch_spyre/_monkey_patch.py#L431) | guard for reusing a `nested_compile_region` subgraph | the subgraph is not reused across sizes |
| [_monkey_patch.py:1131](torch_spyre/_monkey_patch.py#L1131) | the on-disk compile cache key | not a guard, so nothing fails. A new process at a new size misses the cache and pays a full compile |

#### Reproducer

`guards/repro_layout_guard.py`. Uses `x * 2` rather than `for_each_tile`, so the
divisibility bug above does not interfere and every size compiles.

```python
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
        # #2434: the runtime does not yet pass the extra size argument a
        # dynamic graph expects. The compile has already happened by this
        # point, which is what we are measuring.
        print(f"  runtime: {str(e).splitlines()[0][:70]}")
```

Run with `TORCH_LOGS="recompiles"`.

#### Result

All four sizes are inside `[64, 512]`, so Dynamo's range guard passes every
time. Three recompiles happen anyway, and every rejection names only the layout
guard:

```text
S=128  device_size=[2, 128, 64]  stride_map=[64, 128, 1]
S=256  device_size=[2, 256, 64]  stride_map=[64, 128, 1]
  Recompiling function f
    triggered by the following guard failure(s):
    - 0/0: SpyreTensorLayout(x) == SpyreTensorLayout(device_size=[2, 128, 64], ...)
S=320  device_size=[2, 320, 64]  stride_map=[64, 128, 1]
  Recompiling function f
    triggered by the following guard failure(s):
    - 0/1: SpyreTensorLayout(x) == SpyreTensorLayout(device_size=[2, 256, 64], ...)
    - 0/0: SpyreTensorLayout(x) == SpyreTensorLayout(device_size=[2, 128, 64], ...)
S=448  device_size=[2, 448, 64]  stride_map=[64, 128, 1]
  Recompiling function f
    triggered by the following guard failure(s):
    - 0/2: SpyreTensorLayout(x) == SpyreTensorLayout(device_size=[2, 320, 64], ...)
    - 0/1: SpyreTensorLayout(x) == SpyreTensorLayout(device_size=[2, 256, 64], ...)
    - 0/0: SpyreTensorLayout(x) == SpyreTensorLayout(device_size=[2, 128, 64], ...)
```

Four sizes, four compiles. `mark_dynamic` bought nothing.

`stride_map` is identical in every line. Only `device_size[1]` differs, which
is the row count. So the fix is to skip that one entry and compare everything
else exactly.



### 2. Validate the declaration

This guard checks that the declared range is usable with the declared
granularity:

| Check | Example that fails, with `G = 64` |
|---|---|
| `min % G == 0` and `max % G == 0` | `max = 500` |
| `min >= 2*G` | `min = 64` |

#### Why `min >= 2*G`

If `min == G`, then `S // G` is 1 for the smallest sizes in the range, and
PyTorch cannot compile one binary that covers both a dimension of size 1 and one
of any other size.

#### Where it runs

In `.to()`, when the dimension is declared, before anything is compiled. Only
the declared `min`, `max` and `G` are checked here. The runtime size is guard
3's job.

If a check fails, nothing is compiled. The user gets an immediate error that
names the failing value and what to change, for example:

```text
dim 0: min=64 is less than 2*granularity (128). Use min >= 128.
```

### 3. Check the size on every call

`min <= S <= max` and `S % G == 0`, checked every runtime call, with a clear
refusal naming the nearest usable sizes.


On every call, Dynamo first checks its guards, the assumptions recorded at
compile time. Only if all of them pass does the compiled kernel run.

Dynamo's guards already cover both conditions:

| Condition | Where Dynamo's guard comes from |
|---|---|
| `min <= S <= max` | the range given to `mark_dynamic` |
| `S % G == 0` | the divisibility check in `for_each_tile` |

So a bad size always fails a Dynamo guard first. Dynamo's response is to
recompile, not to report an error. 

#### Where it runs

Between Dynamo's guard failure and the recompile. When no compiled entry
matches, Dynamo calls one function to start a new compile
(`torch._dynamo.eval_frame._callback_from_stance`). We wrap it, the same way
`_monkey_patch.py` already wraps `TENSOR_MATCH`.

The wrapper reads `min`, `max` and `G` from the real input tensors, checks `S`,
and if it is bad raises a clear error. The recompile never starts:

```text
dim 0: S=200 is not a multiple of granularity 64. Declared range 128..512.
Nearest valid sizes: 192, 256.
```

The first call always reaches the wrapper, because nothing is compiled yet, so
the first size is checked too. After that, a valid size passes Dynamo's guards
and never reaches the wrapper, so normal calls pay nothing.

**Note:** this design relies on Dynamo holding the `S % G == 0` guard from
`for_each_tile`. How the `ConstraintViolationError` bug at the start of this
doc is resolved may change that, and this guard will be revisited once the
fix is chosen.



### 4. Tied inputs must share one declaration

When two inputs must have the same size along the dynamic dimension, as in
`x + y`, a `torch._check(x.size(0) == y.size(0))` ties their sizes together.
PyTorch then uses one symbol for both, so there is one loop. 

`torch._check` also adds a Dynamo guard, but that guard only checks that the
two sizes are equal on each call. It never compares the declarations.

So we can introduce a guard to compare the declarations and ensure the symbols for the two inputs match

#### Where it runs

At compile time, immediately before the `torch._check` is issued. It reads both
declarations from the side table written by `.to()` and compares `min`, `max`
and `G`. If they match, the `torch._check` goes ahead. If not, the user gets a
clear error:

```text
x dim 0 and y dim 0 must be the same size, but are declared differently:
max 512 vs 256, granularity 64 vs 128. Declare them identically.
```

### 5. Make the `ConstraintViolationError` message readable

Not strictly necessary. It adds no check . The error already fires; this only
replaces the message, which today is a list of sympy expressions with nothing
saying which number in the declaration was wrong.

### 6. Alarm on a recompile caused by the dynamic dimension



Sometimes a valid size still triggers a recompile, because some unrelated guard
on Dynamo's list fails. The layout guard in section 1 is an example: it
recompiled at every new size. The answers stay correct, only slower, so nothing
reports that dynamic shapes have stopped working.

#### Where it runs

In the same wrapper as guard 3. If the wrapper runs on a later call and guard 3
finds the size valid, the recompile should not have been needed. We ask Dynamo
which guard failed, the same way the `fail_on_recompile` stance reports it, and
warn:

```text
Recompiling for x dim 0 = 320, which is a valid size.
Dynamic shapes should have reused the existing compile.
Guard that failed: SpyreTensorLayout(x) == SpyreTensorLayout(device_size=[2, 256, 64], ...)
```

#### Warning, or error with a switch

A warning by default. In production the answers are still correct, and taking a
running model down over a slowdown is worse than a loud warning.

