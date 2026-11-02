-- lua/metrics/init.lua -- metrics 入口
-- torchmetrics 风格指标库(paddle-lua 生态,纯 Lua 零编译)。
--   local metrics = require "metrics"
--   local acc = metrics.Accuracy()
--   acc:update(preds, target); acc:compute()
--   -- 或一次性:acc:score(preds, target)
--   local mc = metrics.MetricCollection{ acc = metrics.Accuracy(),
--                                        mae = metrics.MAE() }
-- 设计:paddle-lua docs/migration/plan/overview.md §6.2。

local M = {}

-- base
M.Metric = require "metrics.base"

-- collection
M.MetricCollection = require "metrics.collection"

-- metric classes(first batch)
local cls = require "metrics.classification"
M.Accuracy = cls.Accuracy
M.Precision = cls.Precision
M.Recall = cls.Recall
M.F1 = cls.F1
local reg = require "metrics.regression"
M.MAE = reg.MAE
M.MSE = reg.MSE
M.RMSE = reg.RMSE
M.R2 = reg.R2

return M