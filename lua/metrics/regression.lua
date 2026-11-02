-- lua/metrics/regression.lua -- 回归指标(首批)
-- MAE / MSE / RMSE / R2。语义照 torchmetrics.regression:
--   MAE  = Σ|y-ŷ| / n
--   MSE  = Σ(y-ŷ)² / n
--   RMSE = sqrt(MSE)
--   R2   = 1 - ss_res / ss_tot(增量累计:sum_error² 归心,R2 对全量数据精确)
-- 状态累加器全部 float32 tensor,compute 除零用 eps 保护。

local class = require "pl.class"
local paddle = require "paddle"
local Metric = require "metrics.base"

local mean_abs_error = class(Metric)

function mean_abs_error:_init(config)
  Metric._init(self, config)
  self:add_state{ "sum_abs", paddle.zeros{ 1 }, dist_reduce_fx = "sum" }
  self:add_state{ "count", paddle.zeros{ 1 }, dist_reduce_fx = "sum" }
end

function mean_abs_error:update(preds, target)
  self:_check_tensors(preds, target)
  local diff = paddle.subtract{ preds, target }
  self.sum_abs = self.sum_abs + paddle.sum{ paddle.abs{ diff } }
  self.count = self.count + self:_numel_as_tensor(preds)
end

function mean_abs_error:compute()
  return paddle.divide{ self.sum_abs, self.count }
end

local mean_squared_error = class(Metric)

function mean_squared_error:_init(config)
  Metric._init(self, config)
  self:add_state{ "sum_sq", paddle.zeros{ 1 }, dist_reduce_fx = "sum" }
  self:add_state{ "count", paddle.zeros{ 1 }, dist_reduce_fx = "sum" }
end

function mean_squared_error:update(preds, target)
  self:_check_tensors(preds, target)
  local diff = paddle.subtract{ preds, target }
  self.sum_sq = self.sum_sq + paddle.sum{ paddle.square{ diff } }
  self.count = self.count + self:_numel_as_tensor(preds)
end

function mean_squared_error:compute()
  return paddle.divide{ self.sum_sq, self.count }
end

local root_mean_squared_error = class(Metric)
-- RMSE 直接组合 MSE 的状态(MSE 已按全量累计),避免二次采样差。
function root_mean_squared_error:_init(config)
  Metric._init(self, config)
  self._mse = mean_squared_error()
  self:add_state{ "sum_sq", paddle.zeros{ 1 }, dist_reduce_fx = "sum" }
  self:add_state{ "count", paddle.zeros{ 1 }, dist_reduce_fx = "sum" }
end

function root_mean_squared_error:update(preds, target)
  self:_check_tensors(preds, target)
  local diff = paddle.subtract{ preds, target }
  self.sum_sq = self.sum_sq + paddle.sum{ paddle.square{ diff } }
  self.count = self.count + self:_numel_as_tensor(preds)
end

function root_mean_squared_error:compute()
  return paddle.sqrt{ paddle.divide{ self.sum_sq, self.count } }
end

-- R2 (coef of determination): 1 - ss_res/ss_tot
--   ss_res = Σ(y-ŷ)² 的归心(centered)形式:Σe² - (Σe)²/n
--   ss_tot = Σy²  - (Σy)²/n
-- 增量状态:sum_error, sum_sq_error, sum_obs, sum_sq_obs, n
local r2_score = class(Metric)

function r2_score:_init(config)
  Metric._init(self, config)
  self:add_state{ "sum_error", paddle.zeros{ 1 }, dist_reduce_fx = "sum" }
  self:add_state{ "sum_sq_error", paddle.zeros{ 1 }, dist_reduce_fx = "sum" }
  self:add_state{ "sum_obs", paddle.zeros{ 1 }, dist_reduce_fx = "sum" }
  self:add_state{ "sum_sq_obs", paddle.zeros{ 1 }, dist_reduce_fx = "sum" }
  self:add_state{ "count", paddle.zeros{ 1 }, dist_reduce_fx = "sum" }
end

function r2_score:update(preds, target)
  self:_check_tensors(preds, target)
  local err = paddle.subtract{ preds, target }
  self.sum_error = self.sum_error + paddle.sum{ err }
  self.sum_sq_error = self.sum_sq_error + paddle.sum{ paddle.square{ err } }
  self.sum_obs = self.sum_obs + paddle.sum{ target }
  self.sum_sq_obs = self.sum_sq_obs + paddle.sum{ paddle.square{ target } }
  self.count = self.count + self:_numel_as_tensor(preds)
end

function r2_score:compute()
  local n = self.count
  -- 归心:ss = Σx² - (Σx)²/n;整体乘 n 变成整数运算(n*ss_std,
  -- R2 的 n 因子上下抵消——中心化残差语义,torchmetrics center=True)。
  local ss_res = self.sum_sq_error * n - self.sum_error * self.sum_error
  local ss_tot = self.sum_sq_obs * n - self.sum_obs * self.sum_obs
  local den = paddle.maximum{ ss_tot, paddle.full{ { 1 }, 1e-7 } }
  return paddle.subtract{ paddle.ones{ 1 }, paddle.divide{ ss_res, den } }
end

-- ---------------------------------------------------------------------
-- PearsonCorrCoef -- Pearson 相关系数(torchmetrics 增量公式):
--   r = (nΣxy - ΣxΣy) / sqrt((nΣx²-(Σx)²) · (nΣy²-(Σy)²))
-- ---------------------------------------------------------------------
local pearson_corr_coef = class(Metric)

function pearson_corr_coef:_init(config)
  Metric._init(self, config)
  self:add_state{ "sum_x", paddle.zeros{ 1 }, dist_reduce_fx = "sum" }
  self:add_state{ "sum_y", paddle.zeros{ 1 }, dist_reduce_fx = "sum" }
  self:add_state{ "sum_x2", paddle.zeros{ 1 }, dist_reduce_fx = "sum" }
  self:add_state{ "sum_y2", paddle.zeros{ 1 }, dist_reduce_fx = "sum" }
  self:add_state{ "sum_xy", paddle.zeros{ 1 }, dist_reduce_fx = "sum" }
  self:add_state{ "count", paddle.zeros{ 1 }, dist_reduce_fx = "sum" }
end

function pearson_corr_coef:update(preds, target)
  self:_check_tensors(preds, target)
  self.sum_x = self.sum_x + paddle.sum{ preds }
  self.sum_y = self.sum_y + paddle.sum{ target }
  self.sum_x2 = self.sum_x2 + paddle.sum{ preds * preds }
  self.sum_y2 = self.sum_y2 + paddle.sum{ target * target }
  self.sum_xy = self.sum_xy + paddle.sum{ preds * target }
  self.count = self.count + self:_numel_as_tensor(preds)
end

function pearson_corr_coef:compute()
  local n = self.count
  local num = self.sum_xy * n - self.sum_x * self.sum_y
  local vx = self.sum_x2 * n - self.sum_x * self.sum_x
  local vy = self.sum_y2 * n - self.sum_y * self.sum_y
  local den = paddle.sqrt{ paddle.maximum{ vx * vy, paddle.full{ { 1 }, 1e-12 } } }
  return paddle.divide{ num, den }
end

return {
  MAE = mean_abs_error,
  MSE = mean_squared_error,
  RMSE = root_mean_squared_error,
  R2 = r2_score,
  PearsonCorrCoef = pearson_corr_coef,
}