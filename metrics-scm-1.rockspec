package = "metrics"
version = "scm-1"

source = {
    url = "git+https://github.com/PlumBlossomMaid/metrics-lua.git",
    branch = "main",
}

description = {
    summary = "torchmetrics-style metrics (add_state state machine) for paddle-lua",
    detailed = [[
        metrics-lua is a pure-Lua, zero-compile metric library on top of
        paddle-lua's public API (paddle.zeros/cast/sum/..., overview.md
        sec.1 of paddle-lua: ecosystem repos only call the public API).
        Core is the add_state declarative state machine: reset / state
        accumulation / compute fully automatic (torchmetrics / PaddleMetrics
        design, paddle-lua overview.md sec.6.2).

        First batch: classification Accuracy/Precision/Recall/F1 +
        regression MAE/MSE/RMSE/R2 + MetricCollection (the Ocean
        log_dict() integration point).
    ]],
    homepage = "https://github.com/PlumBlossomMaid/metrics-lua",
    license = "Apache-2.0",
}

-- Hard deps: Penlight (pl.class/pl.List, the shared ecosystem base --
-- paddle-lua D34: NEVER vendor it) and argrule (argument parsing, D31).
-- paddle-lua itself is required at runtime by every metric (it computes
-- on paddle.Tensor), declared as a dependency for luarocks installs.
dependencies = {
    "lua >= 5.1",
    "penlight >= 1.13",
    "argrule >= 0.1, < 0.2",
    "paddle-lua >= 0.1",
}

build = {
    type = "builtin",
    modules = {
        ["metrics"] = "lua/metrics/init.lua",
        ["metrics.base"] = "lua/metrics/base.lua",
        ["metrics.collection"] = "lua/metrics/collection.lua",
        ["metrics.classification"] = "lua/metrics/classification.lua",
        ["metrics.regression"] = "lua/metrics/regression.lua",
        ["metrics.aggregate"] = "lua/metrics/aggregate.lua",
        ["metrics.image"] = "lua/metrics/image.lua",
        ["metrics.retrieval"] = "lua/metrics/retrieval.lua",
        ["metrics.text"] = "lua/metrics/text.lua",
    },
}