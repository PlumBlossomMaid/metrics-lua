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
-- Accuracy{ multiclass = true, num_classes = C } -- logits [n,C] →
-- argmax axis=1 → 1-based 类标与 target 比较(torchmetrics multi-class)。
-- ---------------------------------------------------------------------
local Accuracy = class(Metric)

function Accuracy:_init(config)
  Metric._init(self, config)
  self.exact_count = config and config.exact_count  -- 默认 false:preds 阈值化
  self.multiclass = (config and config.multiclass) or false
  self.num_classes = config and config.num_classes
  -- 标量常量转 [1] tensor:顶层层 first-batch 比较算子只收 Tensor 对
  -- (greater_than{x,y} / equal{x,y},标量操作数是后续梯队)
  self._th = paddle.full{ { 1 }, (config and config.threshold) or 0.5 }
  self:add_state{ "correct", paddle.zeros{ 1 }, dist_reduce_fx = "sum" }
  self:add_state{ "total", paddle.zeros{ 1 }, dist_reduce_fx = "sum" }
end

function Accuracy:update(preds, target)
  self:_check_tensors(preds, target)
  local pred, cmp
  if self.multiclass then
    -- logits [n,C] → 每行 argmax(类别在第 2 维 = 1-based axis 2);索引
    -- 在 C ABI 层 +1(1-based 类标,index_semantics);equal 要求同 dtype
    pred = paddle.cast{ paddle.argmax{ preds, axis = 2 }, "int32" }
    cmp = paddle.cast{ target, "int32" }
  elseif self.exact_count then
    pred = preds
    cmp = target
  else
    -- 概率/logits → 二值:preds > threshold;精确 label 相等由 exact_count 走
    pred = paddle.cast{ paddle.greater_than{ preds, self._th }, "float32" }
    cmp = target
  end
  local hit = paddle.cast{ paddle.equal{ pred, cmp }, "float32" }
  self.correct = self.correct + paddle.sum{ hit }
  -- 样本数 = 比较后的 pred numel(multiclass 下 preds 是 [n,C] logits,
  -- 逐元素 numel 会把类别数也算进去 —— torchmetrics 用样本数)
  self.total = self.total + self:_numel_as_tensor(pred)
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

-- ---------------------------------------------------------------------
-- ConfusionMatrix{num_classes} -- [C,C] 混淆矩阵。preds/target 是
-- 1-based 类标(多类 logits 先 argmax +1);每批按 (pred=i, target=j)
-- 对计数累加进 matrix(逐对 sum,免 matmul/unsqueeze 依赖)。
-- compute() 返回累加后的 [C,C] tensor。
-- ---------------------------------------------------------------------
local ConfusionMatrix = class(Metric)

function ConfusionMatrix:_init(config)
  Metric._init(self, config)
  local nc = config and config.num_classes
  if not nc then
    error("metrics.ConfusionMatrix: num_classes is required", 3)
  end
  self.num_classes = nc
  self:add_state{ "matrix", paddle.zeros{ nc, nc }, dist_reduce_fx = "sum" }
end

function ConfusionMatrix:update(preds, target)
  self:_check_tensors(preds, target)
  local C = self.num_classes
  local rows = {}
  for i = 1, C do
    local pi = paddle.cast{ paddle.equal{ preds, paddle.full{ { 1 }, i } },
                            "float32" }
    local cols = {}
    for j = 1, C do
      local tj = paddle.cast{ paddle.equal{ target, paddle.full{ { 1 }, j } },
                              "float32" }
      cols[j] = paddle.sum{ pi * tj }  -- 0-d 计数
    end
    rows[i] = paddle.stack{ x = cols, axis = 1 }  -- [C]
  end
  local batch = paddle.stack{ x = rows, axis = 1 }  -- [C,C]
  self.matrix = self.matrix + batch
end

function ConfusionMatrix:compute()
  return self.matrix
end

return {
  Accuracy = Accuracy,
  Precision = Precision,
  Recall = Recall,
  F1 = F1,
  ConfusionMatrix = ConfusionMatrix,
}