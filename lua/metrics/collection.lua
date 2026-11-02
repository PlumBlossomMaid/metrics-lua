-- lua/metrics/collection.lua -- MetricCollection
-- 一组指标的容器:统一 update/reset/compute,返回 {name=value};是 Ocean
-- 侧 `self:log_dict()` 的对接点(paddle-lua overview.md §6.2)。
-- 构造:MetricCollection{ acc = Accuracy{}, mae = MAE{} }

local class = require "pl.class"

local MetricCollection = class()

function MetricCollection:_init(metrics)
  metrics = metrics or {}
  self._metrics = {}
  self._order = {}
  for name, m in pairs(metrics) do
    if type(name) ~= "string" then
      error("metrics.MetricCollection: metric names must be strings", 3)
    end
    self._metrics[name] = m
    self._order[#self._order + 1] = name
  end
  -- Lua 表不保留键序(pairs 无序);keys()/compute() 按名称排序确定性。
  table.sort(self._order)
end

-- update(...) 广播到每个指标。各指标的 update 签名必须一致(通常都是
-- (preds, target));torchmetrics 的分发逻辑精简为本约束。
function MetricCollection:update(...)
  for _, m in pairs(self._metrics) do
    m:update(...)
  end
  return self
end

function MetricCollection:reset()
  for _, m in pairs(self._metrics) do
    m:reset()
  end
  return self
end

-- compute():{ name = 指标值 tensor } 或 { name = 表 } —— 保持原子值
-- 形态,Ocean log_dict 直接消费。
function MetricCollection:compute()
  local out = {}
  for _, name in ipairs(self._order) do
    out[name] = self._metrics[name]:compute()
  end
  return out
end

-- log_dict 别名(与 Ocean `self:log{}` 语义对齐;名字保留为文档契约)。
function MetricCollection:log_dict()
  return self:compute()
end

function MetricCollection:keys()
  local ks = {}
  for _, name in ipairs(self._order) do ks[#ks + 1] = name end
  return ks
end

function MetricCollection:values()
  local vs = {}
  for _, name in ipairs(self._order) do vs[#vs + 1] = self._metrics[name] end
  return vs
end

function MetricCollection:get(name)
  return self._metrics[name]
end

return MetricCollection