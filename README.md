# metrics-lua

torchmetrics-style metrics for [paddle-lua](https://github.com/PlumBlossomMaid/Paddle-Lua) —
pure Lua 5.1, zero native compile, one `add_state` declarative state machine.

```lua
local metrics = require "metrics"

local acc = metrics.Accuracy()
acc:update(preds, target)          -- accumulate over batches
local v = acc:compute()            -- score tensor

-- or one-shot:
local v2 = acc:score(preds, target)

-- group metrics (the Ocean log_dict() integration point):
local mc = metrics.MetricCollection{
  acc = metrics.Accuracy(),
  mae = metrics.MAE(),
}
mc:update(preds, target)
local log = mc:log_dict()          -- { acc = tensor, mae = tensor }
```

## First batch

| family | metrics |
|---|---|
| classification | `Accuracy` `Precision` `Recall` `F1`(binary, `threshold` config) |
| regression | `MAE` `MSE` `RMSE` `R2` |
| base | `Metric`(add_state/reset/compute/score)+ `MetricCollection`(update/reset/log_dict/keys/values) |

Semantics follow torchmetrics.classification / torchmetrics.regression
(binary accuracy/precision/recall are thresholded comparisons; R2 uses the
centered-residual incremental formula, `center=True`).

## Design

`add_state{name, default_tensor, dist_reduce_fx="sum"}` registers an
accumulator under `self.<name>`; `update()` re-assigns it, `compute()`
reads states, `reset()` restores defaults — same shape as torchmetrics /
PaddleMetrics. `dist_reduce_fx` is recorded for the future distributed
sync; accumulation is float32 throughout.

Consumes only paddle-lua's **public API** (`paddle.zeros/cast/sum/equal/
greater_than/...`) — ecosystem repos must not reach into the C ABI
(paddle-lua overview.md §1).

## Dependencies (rocks)

- `penlight`(pl.class, shared ecosystem base — never vendored)
- `argrule`(argument parsing)
- `paddle-lua`(runtime)

## Run the smoke

From the paddle-lua repo root(so `./build-test/paddle_core.dll`
resolves)with the usual paddle runtime PATH:

```bash
luajit E:\code\metrics-lua\tools\smoke\metrics_smoke.lua
# SMOKE PASS: metrics-lua first batch OK (18 assertions)
```

## License

Apache-2.0.