-- lua/metrics/aggregate.lua -- 通用聚合指标(torchmetrics.aggregation)
-- MeanMetric: 逐步累积 sum/count,compute = sum/count
-- SumMetric:  逐步累积 sum,compute = sum
-- CatMetric:  拼接 —— 需要 concat,顶层层尚未暴露;留待后续。

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

return {
  MeanMetric = mean_metric,
  SumMetric = sum_metric,
}