-- lua/metrics/classification.lua -- 分类指标(首批)
-- Accuracy(元素级,已对齐 preds/target)+ Precision/Recall/F1
-- (binary,threshold=t 判正)。全部在 paddle Tensor 上,paddle-lua 公开 API。
--
-- 语义:torchmetrics.classification(binary Precision/Recall/F1,
-- 阈值化 + TP/FP/FN 计数);single-label 多类 v1 后续随 argmax 顶层层。

local class = require "pl.class"
local paddle = require "paddle"
local Metric = require "metrics.base"

-- ---------------------------------------------------------------------
-- Accuracy{threshold} -- preds==target 的占比(元素级;多类用已对齐 label)
-- ---------------------------------------------------------------------
local Accuracy = class(Metric)

function Accuracy:_init(config)
  Metric._init(self, config)
  self.exact_count = config and config.exact_count  -- 默认 false:preds 阈值化
  -- 标量常量转 [1] tensor:顶层层 first-batch 比较算子只收 Tensor 对
  -- (greater_than{x,y} / equal{x,y},标量操作数是后续梯队)
  self._th = paddle.full{ { 1 }, (config and config.threshold) or 0.5 }
  self:add_state{ "correct", paddle.zeros{ 1 }, dist_reduce_fx = "sum" }
  self:add_state{ "total", paddle.zeros{ 1 }, dist_reduce_fx = "sum" }
end

function Accuracy:update(preds, target)
  self:_check_tensors(preds, target)
  local pred
  if self.exact_count then
    pred = preds
  else
    -- 概率/logits → 二值:preds > threshold;精确 label 相等由 exact_count 走
    pred = paddle.cast{ paddle.greater_than{ preds, self._th },
                        "float32" }
  end
  local hit = paddle.cast{ paddle.equal{ pred, target }, "float32" }
  self.correct = self.correct + paddle.sum{ hit }
  self.total = self.total + self:_numel_as_tensor(preds)
end

function Accuracy:compute()
  return paddle.divide{ self.correct, self.total }
end

-- ---------------------------------------------------------------------
-- 二值混淆统计基类:tp/fp/fn(binary,threshold 判正)
-- ---------------------------------------------------------------------
local BinaryMetrics = class(Metric)

function BinaryMetrics:_init(config)
  Metric._init(self, config)
  self._th = paddle.full{ { 1 }, (config and config.threshold) or 0.5 }
  self:add_state{ "tp", paddle.zeros{ 1 }, dist_reduce_fx = "sum" }
  self:add_state{ "fp", paddle.zeros{ 1 }, dist_reduce_fx = "sum" }
  self:add_state{ "fn", paddle.zeros{ 1 }, dist_reduce_fx = "sum" }
end

function BinaryMetrics:update(preds, target)
  self:_check_tensors(preds, target)
  -- pred 判正 (binary: > threshold),target 判正 (=1)
  local pos = paddle.cast{ paddle.greater_than{ preds, self._th },
                           "float32" }
  local ones = paddle.full{ pos:shape(), 1.0 }
  local neg = paddle.subtract{ ones, pos }
  local one = paddle.full{ { 1 }, 1.0 }
  local tpos = paddle.cast{ paddle.equal{ target, one }, "float32" }
  local tneg = paddle.subtract{ ones, tpos }
  -- 元方法 * 收双 tensor(paddle 顶层层首批无 multiply,__mul 直通 pd_op)
  self.tp = self.tp + paddle.sum{ pos * tpos }
  self.fp = self.fp + paddle.sum{ pos * tneg }
  self.fn = self.fn + paddle.sum{ neg * tpos }
end

-- 分母取 max(denom, eps) 防除零;返回(float, 分母 0 无意义)
local function safe_div(num, denom, eps)
  eps = eps or 1e-7
  local den = paddle.maximum{ denom, paddle.full{ { 1 }, eps } }
  return paddle.divide{ num, den }
end

-- Precision: TP/(TP+FP)
local Precision = class(BinaryMetrics)
function Precision:compute()
  return safe_div(self.tp, self.tp + self.fp)
end

-- Recall: TP/(TP+FN)
local Recall = class(BinaryMetrics)
function Recall:compute()
  return safe_div(self.tp, self.tp + self.fn)
end

-- F1: 2TP/(2TP+FP+FN)
local F1 = class(BinaryMetrics)
function F1:compute()
  local two_tp = self.tp + self.tp
  return safe_div(two_tp, two_tp + self.fp + self.fn)
end

return {
  Accuracy = Accuracy,
  Precision = Precision,
  Recall = Recall,
  F1 = F1,
}