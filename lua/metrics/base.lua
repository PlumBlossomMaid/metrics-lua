-- lua/metrics/base.lua -- Metric 基类 + add_state 声明式状态机
--
-- torchmetrics / PaddleMetrics 设计(paddle-lua overview.md §6.2):
--   1. 子类 _init 里 add_state 登记状态(默认值 tensor)
--   2. update(...) 里以"重新赋值"语义更新状态(self.correct = self.correct + x)
--   3. compute() 从状态算指标值
-- reset() 自动把所有状态重赋默认值;MetricCollection 统一 update/reset。
--
-- 纯 Lua 5.1 + pl.class(生态基座,D34 不 vendor);计算全部走 paddle-lua
-- 公开 API(顶层层 paddle.zeros/cast/sum/equal/greater_than/...)。

local class = require "pl.class"
local paddle = require "paddle"

local Metric = class()

function Metric:_init(config)
  config = config or {}
  self._states = {}                 -- name -> { default=..., reduce=... }
  self._class_module = config._module  -- 子类归属模块(错误消息用)
end

-- add_state{name, default, dist_reduce_fx="sum"} -- torchmetrics 语义。
-- 把 name 登记并实例化成 self.<name> = default(plain 字段,可读写);
-- update 重新赋值,compute 读。reset 重赋 default。
function Metric:add_state(...)
  local name, default, opts
  local arg = ...
  if type(arg) == "table" then
    -- add_state{"correct", paddle.zeros{1}, dist_reduce_fx="sum"}
    name, default = arg[1], arg[2]
    if arg.dist_reduce_fx ~= nil then opts = { dist_reduce_fx = arg.dist_reduce_fx } end
  else
    -- add_state("correct", paddle.zeros{1}, {dist_reduce_fx="sum"})
    name, default, opts = arg, select(2, ...), select(3, ...)
  end
  if type(name) ~= "string" or default == nil then
    error("metrics: add_state needs {name, default[, dist_reduce_fx]}", 3)
  end
  local reduce = (opts and opts.dist_reduce_fx) or "sum"
  self._states[name] = { default = default, reduce = reduce }
  self[name] = default
end

-- reset: 全状态重赋默认值(顺序无关)。
function Metric:reset()
  for name, s in pairs(self._states) do
    self[name] = s.default
  end
  return self
end

-- Torchmetrics 核心:重置后回归"干净"状态,可复用同一 metric 实例。
-- 与 reset 等价(本实现无动量/历史残留)。
function Metric:_restart()
  return self:reset()
end

-- 一次性打分(等价 torchmetrics metric(preds, target)):reset → update
-- → compute。显式方法而非 __call —— pl.class 的 __call 是类构造器。
function Metric:score(...)
  self:reset()
  self:update(...)
  return self:compute()
end

-- 子类必须实现
function Metric:update(...)
  error("metrics: " .. tostring(getmetatable(self))
        .. " must implement update(...)", 2)
end

function Metric:compute()
  error("metrics: " .. tostring(getmetatable(self))
        .. " must implement compute()", 2)
end

-- 通用守卫:preds/target 必须是 paddle Tensor(userdata,共享 metatable)。
local probe = paddle.zeros{ 1 }
local TENSOR_MT = getmetatable(probe)
local function is_tensor(v)
  return type(v) == "userdata" and getmetatable(v) == TENSOR_MT
end
function Metric:_check_tensors(...)
  for i = 1, select("#", ...) do
    if not is_tensor((select(i, ...))) then
      error("metrics: argument " .. i .. " is not a paddle Tensor", 3)
    end
  end
end

-- 计数累加:numel -> [1] float32 tensor(状态都是 tensor,加减只收 tensor)。
function Metric:_numel_as_tensor(t)
  return paddle.full{ { 1 }, t:numel() }
end

Metric._is_tensor = is_tensor

return Metric