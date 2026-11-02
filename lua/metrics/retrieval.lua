-- lua/metrics/retrieval.lua -- 检索指标(torchmetrics.retrieval,首批 4)
-- AUROC / AveragePrecision / MRR / NDCG。
--
-- torchmetrics retrieval 语义:update 只替换最新 (preds, target)(非增量),
-- compute 对最新样本算 —— 状态直接存张量,不走 add_state 累加。
--
-- AUC 实现:升序 rank 外积(向量化,所有规模)。rank[i]=#{preds[j]<preds[i]}+1,
--   AUC = (Σ rank_pos − n_pos(n_pos+1)/2) / (n_pos·n_neg)(Mann-Whitney,
--   无 tie 处理 —— tie 时 rank 用 < 计数,torchmetrics 默认对 tie 有专门
--   处理,简化版注明)。
-- 其余三款按降序排列逐项累加 —— retrieval 逐 query 小规模,Lua 循环可接受。

local class = require "pl.class"
local paddle = require "paddle"
local Metric = require "metrics.base"

local RetrievalMetric = class(Metric)

function RetrievalMetric:_init(config)
  Metric._init(self, config)
  self._preds = nil
  self._target = nil
end

function RetrievalMetric:update(preds, target)
  self:_check_tensors(preds, target)
  self._preds = preds
  self._target = target
end

function RetrievalMetric:reset()
  self._preds = nil
  self._target = nil
end

function RetrievalMetric:_require()
  if self._preds == nil then
    error("metrics: " .. tostring(getmetatable(self))
          .. " has no update() data yet", 3)
  end
  return self._preds, self._target
end

-- f32 at byte idx (1-based) of a bytes string
local function f32_at(s, idx)
  local b1, b2, b3, b4 = s:byte(idx, idx + 3)
  local bits = b1 + b2 * 256 + b3 * 65536 + b4 * 16777216
  local sign = bits >= 2147483648 and 1 or 0
  local exp = math.floor(bits / 8388608) % 256
  local mant = bits % 8388608
  if exp == 0 then return 0.0 end
  if exp == 255 then return math.huge end
  local v = (1 + mant / 8388608) * 2 ^ (exp - 127)
  return sign == 1 and -v or v
end

-- tensor scalar -> Lua number(f32)
local function scalar(t)
  return f32_at(t:to_host_bytes(), 1)
end

-- 1-D f32 tensor -> 1-based Lua number 数组
local function to_array(t)
  local s = t:to_host_bytes()
  local n = t:numel()
  local out = {}
  for i = 1, n do out[i] = f32_at(s, (i - 1) * 4 + 1) end
  return out
end

local function as_tensor(v)
  return paddle.full{ { 1 }, v }
end

-- ---------------------------------------------------------------------
-- AUROC(binary):label 1 = 正。无 tie 简化的 rank 公式。
-- ---------------------------------------------------------------------
local AUROC = class(RetrievalMetric)

function AUROC:compute()
  local preds, target = self:_require()
  local n = preds:numel()
  local tf = paddle.cast{ target, "float32" }
  local n_pos = scalar(paddle.sum{ tf })
  local n_neg = n - n_pos
  if n_pos == 0 or n_neg == 0 then
    return paddle.full{ { 1 }, 0 / 0 }  -- NaN(全正/全负,torchmetrics 同)
  end
  -- 降序 rank(无 tie):rank[i] = #{preds[j] > preds[i]} + 1
-- (less_than 外积给出"j 比 i 大"的计数 → 降序秩)。Mann-Whitney 用升序
-- 秩和换算:Σr'_pos = n·n_pos − Σr_pos + n_pos。
local sc2 = paddle.unsqueeze{ preds, axis = 2 }         -- [n,1]
  local lt = paddle.cast{ paddle.less_than{ sc2, preds }, "float32" }
  local rank = paddle.sum{ lt, axis = 2 } + paddle.ones{ n }  -- 降序 rank [n]
  local sum_rank_pos = scalar(paddle.sum{ rank * tf })
  local auc_num = n * n_pos - sum_rank_pos + n_pos
                  - n_pos * (n_pos + 1) / 2
  return as_tensor(auc_num / (n_pos * n_neg))
end

-- ---------------------------------------------------------------------
-- 降序位置集合(preds 从高到低排列的原始下标) —— 供 AP/MRR/NDCG 共享。
-- 返回 { order = {1..n 位置}, ta = target 数组, pa = preds 数组 }
-- ---------------------------------------------------------------------
local function desc_context(preds, target)
  local pa = to_array(preds)
  local ta = to_array(target)
  local n = #pa
  local order = {}
  for i = 1, n do order[i] = i end
  table.sort(order, function(a, b) return pa[a] > pa[b] end)
  return order, ta, n
end

-- ---------------------------------------------------------------------
-- AveragePrecision(binary):AP = (1/n_pos) Σ_{k:rel_k} (TP_k / k)
-- ---------------------------------------------------------------------
local AveragePrecision = class(RetrievalMetric)

function AveragePrecision:compute()
  local preds, target = self:_require()
  local order, ta = desc_context(preds, target)
  local n_pos = 0
  for i = 1, #ta do
    if ta[i] > 0.5 then n_pos = n_pos + 1 end
  end
  if n_pos == 0 then return paddle.full{ { 1 }, 0 / 0 } end  -- NaN
  local tp, ap = 0, 0
  for k = 1, #order do
    if ta[order[k]] > 0.5 then
      tp = tp + 1
      ap = ap + tp / k
    end
  end
  return as_tensor(ap / n_pos)
end

-- ---------------------------------------------------------------------
-- MRR:首个正样本的倒数 rank(降序)
-- ---------------------------------------------------------------------
local MRR = class(RetrievalMetric)

function MRR:compute()
  local preds, target = self:_require()
  local order, ta = desc_context(preds, target)
  for k = 1, #order do
    if ta[order[k]] > 0.5 then
      return as_tensor(1.0 / k)
    end
  end
  return as_tensor(0)  -- 无相关结果
end

-- ---------------------------------------------------------------------
-- NDCG:DCG/IDCG,rel = target 值(binary 0/1 简化)
--   DCG = Σ_k (2^rel_k − 1) / log2(k+1);IDCG 对 target 自身降序
-- ---------------------------------------------------------------------
local NDCG = class(RetrievalMetric)

local function dcg_gain(ta, order)
  local g = 0
  for k = 1, #order do
    local rel = ta[order[k]]
    if rel > 0.5 then
      g = g + 1 / math.log(k + 1, 2)  -- (2^1−1)/log2(k+1)
    end
  end
  return g
end

function NDCG:compute()
  local preds, target = self:_require()
  local order, ta = desc_context(preds, target)
  -- IDCG:target 自身降序
  local n = #ta
  local ideal = {}
  for i = 1, n do ideal[i] = i end
  table.sort(ideal, function(a, b) return ta[a] > ta[b] end)
  local dcg = dcg_gain(ta, order)
  local idcg = dcg_gain(ta, ideal)
  if idcg <= 0 then return as_tensor(0) end
  return as_tensor(dcg / idcg)
end

return {
  AUROC = AUROC,
  AveragePrecision = AveragePrecision,
  MRR = MRR,
  NDCG = NDCG,
}