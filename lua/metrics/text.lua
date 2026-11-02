-- lua/metrics/text.lua -- 文本指标(torchmetrics.text)
-- 首批:Bleu(BLEU score)。纯 Lua 字符串处理,零 paddd 算子依赖
-- (状态是文本列表,不走 add_state tensor 机制 —— 与 CatMetric 同款:
-- 累积表在 _init 新建,reset 整表替换)。
--
-- torchmetrics 语义(text/bleu.py):
--   update(target, preds):target/preds 是句子字符串或字符串表;
--     多次 update 累积句子对(全局语料统计)。
--   compute():语料级 BLEU —— n-gram precision 用所有句子的全局计数
--     (clip 到参考最大值),BP 用总长度。
--   n_gram 默认 4,weight 均匀 1/n_gram。
--   smooth=false(默认):任一阶 matches==0 → 整体 BLEU=0(torchmetrics
--     no-smoothing 语义);smooth=true:0 match 阶给 1/denom
--     (Chen & Cherry method 1,与 nltk smoothing_function method1 一致)。
--
-- 纯 Lua 5.1 + pl.class;不 require paddle(文本指标无 tensor)。

local class = require "pl.class"
local Metric = require "metrics.base"

local Bleu = class(Metric)

function Bleu:_init(config)
  Metric._init(self, config)
  config = config or {}
  self.n_gram = config.n_gram or 4
  self.smooth = config.smooth or false
  self._targets = {}
  self._preds = {}
end

-- update(target, preds):target/preds = 句子(string)或 {string,...}。
-- 一一对应;次数不等以少者为准。
function Bleu:update(target, preds)
  local ts = target
  local ps = preds
  if type(target) == "string" then ts = { target } end
  if type(preds) == "string" then ps = { preds } end
  if type(ts) ~= "table" or type(ps) ~= "table" then
    error("metrics: Bleu:update needs string or string-list (target, preds)", 2)
  end
  local n = math.min(#ts, #ps)
  for i = 1, n do
    if type(ts[i]) ~= "string" or type(ps[i]) ~= "string" then
      error(("metrics: Bleu:update element %d is not a string"):format(i), 2)
    end
  end
  for i = 1, n do
    self._targets[#self._targets + 1] = ts[i]
    self._preds[#self._preds + 1] = ps[i]
  end
end

-- 空格分词(标点也当独立 token?torchmetrics 默认用 whitespace tokenizer
-- 且保留标点 —— 这里纯空格切分,镜像 torchmetrics TextTokenizer 的
-- default 行为)。
local function tokenize(s)
  local out = {}
  for w in s:gmatch("%S+") do
    out[#out + 1] = w
  end
  return out
end

-- 句子 n-gram 统计:{ gram = count }
local function ngrams(tokens, n)
  local g = {}
  for i = 1, #tokens - n + 1 do
    local key = table.concat(tokens, " ", i, i + n - 1)
    g[key] = (g[key] or 0) + 1
  end
  return g
end

-- 语料级统计:对所有句子对累积
-- 返回 ref_len / hyp_len / per-order (matches, hyp_total)
function Bleu:_corpus_stats()
  local ref_len = 0
  local hyp_len = 0
  -- 每阶:matches(全局 clip 和)与 hyp n-gram 总数
  local matches = {}
  local hyp_total = {}
  local ref_counts = {}   -- 每阶:参考全局计数 { gram = count(所有参考累加) }
  for k = 1, self.n_gram do
    matches[k] = 0
    hyp_total[k] = 0
    ref_counts[k] = {}
  end
  for i = 1, #self._targets do
    -- 每句独立 clip:torchmetrics 对每个 pred 句 vs 对应 target 句
    local rt = tokenize(self._targets[i])
    local rh = tokenize(self._preds[i])
    ref_len = ref_len + #rt
    hyp_len = hyp_len + #rh
    for k = 1, self.n_gram do
      local tgrams = ngrams(rt, k)
      local hgrams = ngrams(rh, k)
      for g, c in pairs(hgrams) do
        local tc = tgrams[g] or 0
        local cl = math.min(c, tc)          -- per-sentence clip
        matches[k] = matches[k] + cl
        hyp_total[k] = hyp_total[k] + c
      end
    end
  end
  return ref_len, hyp_len, matches, hyp_total
end

-- compute():语料级 BLEU。
-- BLEU = BP * exp( Σ_k w_k * ln(p_k) ),p_k = matches_k / hyp_total_k
-- smooth=false:任一 p_k == 0 → 0.0
-- smooth=true:p_k == 0 的阶用 1/hyp_total_k(Chen & Cherry method 1)
function Bleu:compute()
  local n = #self._targets
  if n == 0 then
    error("metrics: Bleu:update() must be called before compute()", 2)
  end
  local ref_len, hyp_len, matches, hyp_total = self:_corpus_stats()

  -- 平滑前先检查 0 match
  local any_zero = false
  local precisions = {}
  for k = 1, self.n_gram do
    local m, d = matches[k], hyp_total[k]
    if d == 0 then d = 1 end    -- 该阶无候选 n-gram:视为 0(或平滑)
    if m == 0 then
      if self.smooth then
        precisions[k] = 1 / d  -- method 1: 假设 1 个命中
      else
        any_zero = true
      end
    else
      precisions[k] = m / d
    end
  end
  if any_zero then return 0.0 end

  -- brevity penalty(c < r 时惩罚)
  local bp = 1.0
  if hyp_len < ref_len then
    bp = math.exp(1 - ref_len / hyp_len)
  end

  local log_sum = 0.0
  local w = 1 / self.n_gram
  for k = 1, self.n_gram do
    log_sum = log_sum + w * math.log(precisions[k])
  end
  return bp * math.exp(log_sum)
end

function Bleu:reset()
  self._targets = {}
  self._preds = {}
  return self
end

return {
  Bleu = Bleu,
}