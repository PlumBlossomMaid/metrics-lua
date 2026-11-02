-- lua/metrics/aggregate.lua -- 通用聚合指标(torchmetrics.aggregation)
-- MeanMetric: 逐步累积 sum/count,compute = sum/count
-- SumMetric:  逐步累积 sum,compute = sum
-- CatMetric:  收集 tensor 列表,compute 时沿 axis 拼接(torchmetrics 语义)。

local class = require "pl.class"
local paddle = require "paddle"
local Metric = require "metrics.base"

local mean_metric = class(Metric)

function mean_metric:_init(config)
  Metric._init(self, config)
  self:add_state{ "sum", paddle.zeros{ 1 }, dist_reduce_fx = "sum" }
  self:add_state{ "count", paddle.zeros{ 1 }, dist_reduce_fx = "sum" }
end

function mean_metric:update(value)
  -- value: paddle Tensor(任意 shape,全量求均值)或单数值
  if Metric._is_tensor(value) then
    self.sum = self.sum + paddle.sum{ value }
    self.count = self.count + self:_numel_as_tensor(value)
  else
    self.sum = self.sum + paddle.full{ { 1 }, value }
    self.count = self.count + paddle.full{ { 1 }, 1 }
  end
end

function mean_metric:compute()
  return paddle.divide{ self.sum, self.count }
end

local sum_metric = class(Metric)

function sum_metric:_init(config)
  Metric._init(self, config)
  self:add_state{ "sum", paddle.zeros{ 1 }, dist_reduce_fx = "sum" }
end

function sum_metric:update(value)
  local v
  if Metric._is_tensor(value) then
    v = paddle.sum{ value }
  else
    v = paddle.full{ { 1 }, value }
  end
  self.sum = self.sum + v
end

function sum_metric:compute()
  return self.sum
end

-- CatMetric{axis} -- update 追加 tensor 到收集列表,compute 沿 axis 拼接。
-- 1-based axis(默认 1 = 按第一维竖排拼接,对齐 torchmetrics axis=0)。
-- 状态用 Lua 表收集(不能进 add_state:reset 需整表替换,且表默认值
-- 跨实例共享会互相污染 —— 表在 _init 里新建)。
local cat_metric = class(Metric)

function cat_metric:_init(config)
  Metric._init(self, config)
  config = config or {}
  self.axis = config.axis or 1
  self._parts = {}
end

function cat_metric:update(value)
  Metric._check_tensors(value)
  self._parts[#self._parts + 1] = value
end

function cat_metric:compute()
  if #self._parts == 0 then
    error("metrics: CatMetric:update() must be called before compute()", 2)
  end
  return paddle.concat{ x = self._parts, axis = self.axis }
end

function cat_metric:reset()
  self._parts = {}
  return self
end

return {
  MeanMetric = mean_metric,
  SumMetric = sum_metric,
  CatMetric = cat_metric,
}