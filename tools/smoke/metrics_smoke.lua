-- tools/smoke/metrics_smoke.lua -- metrics-lua 首批指标 smoke
-- Hand-computed oracles (torchmetrics semantics):
--   Accuracy  preds=[.8,.2,.9,.1] tgt=[1,0,1,0] thresh .5 -> 4/4 = 1.0
--             +update tgt=[1,1,1,0] -> (4+3)/8 = 0.875
--   P/R/F1    preds=[.9,.8,.2,.1] tgt=[1,0,1,0] -> tp=1 fp=1 fn=1
--             P=0.5 R=0.5 F1=0.5
--   MAE       [3,5] vs [1,3] -> (2+2)/2 = 2.0
--   MSE       same -> 8/2 = 4.0 ;RMSE -> 2.0
--   R2        perfect -> 1.0 ;linear-double [2,4,6] vs [1,2,3] -> 0.0
-- Paths: run from paddle-lua repo root so ./build-test/paddle_core.dll
-- resolves; metrics-lua lua/ + paddle-lua lua/ both on package.path.
package.path = "./lua/?.lua;./lua/?/init.lua;" .. package.path
package.path = "E:\\code\\metrics-lua\\lua/?.lua;E:\\code\\metrics-lua\\lua/?/init.lua;" .. package.path
package.cpath = "./build-test/?.dll;" .. package.cpath

local core = require "paddle_core"
core.init()
local paddle = require "paddle"
local metrics = require "metrics"

local ntests = 0
local function check(cond, msg)
  ntests = ntests + 1
  if not cond then
    error(("assertion #%d failed: %s"):format(ntests, msg), 2)
  end
end

-- f32 LE encode + from_host_bytes (clone of the paddle-lua smokes).
local function from_lua(vals)
  local bytes = {}
  for _, v in ipairs(vals) do
    local sign = v < 0 and 1 or 0
    if v < 0 then v = -v end
    local exp, mant
    if v == 0 then
      exp, mant = 0, 0
    else
      exp = math.floor(math.log(v) / math.log(2)) + 127
      local m = v / 2 ^ (exp - 127)
      mant = math.floor((m - 1) * 8388608 + 0.5)
      if mant >= 8388608 then
        mant = mant - 8388608
        exp = exp + 1
      end
    end
    if exp > 254 then exp = 254 end
    local bits = sign * 2147483648 + exp * 8388608 + mant
    bytes[#bytes + 1] = string.char(bits % 256, math.floor(bits / 256) % 256,
                                    math.floor(bits / 65536) % 256,
                                    math.floor(bits / 16777216) % 256)
  end
  return core.from_host_bytes(table.concat(bytes), { #vals },
                              core.PD_DTYPE_FLOAT32,
                              core.pd_place_new(core.PD_PLACE_CPU, 0))
end

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
local function near(v, target)
  return math.abs(v - target) < 1e-5
end
local function scalar(t)
  return f32_at(t:to_host_bytes(), 1)
end

-- ---- Accuracy ----
local acc = metrics.Accuracy()
local p1 = from_lua { 0.8, 0.2, 0.9, 0.1 }
local t1 = from_lua { 1, 0, 1, 0 }
acc:update(p1, t1)
check(near(scalar(acc:compute()), 1.0), "Accuracy batch1 == 1.0")
acc:update(p1, from_lua { 1, 1, 1, 0 })
check(near(scalar(acc:compute()), 0.875), "Accuracy 2 batches == 7/8")
acc:reset()
acc:update(p1, t1)
check(near(scalar(acc:compute()), 1.0), "Accuracy after reset == 1.0")
-- one-shot score
local acc2 = metrics.Accuracy()
check(near(scalar(acc2:score(p1, t1)), 1.0), "Accuracy:score one-shot == 1.0")

-- ---- Precision / Recall / F1 (binary) ----
local p2 = from_lua { 0.9, 0.8, 0.2, 0.1 }   -- pos = 1,1,0,0
local t2 = from_lua { 1, 0, 1, 0 }           -- tpos= 1,0,1,0 -> tp=1 fp=1 fn=1
local pr = metrics.Precision()
local rc = metrics.Recall()
local f1 = metrics.F1()
for _, m in ipairs({ pr, rc, f1 }) do m:update(p2, t2) end
check(near(scalar(pr:compute()), 0.5), "Precision == tp/(tp+fp) == 0.5")
check(near(scalar(rc:compute()), 0.5), "Recall == tp/(tp+fn) == 0.5")
check(near(scalar(f1:compute()), 0.5), "F1 == 2tp/(2tp+fp+fn) == 0.5")
-- zero-positive guard: all-negative preds -> P and F1 fall back to eps
local pr2 = metrics.Precision()
pr2:update(from_lua { 0.1, 0.2 }, from_lua { 1, 1 })
check(near(scalar(pr2:compute()), 0.0), "Precision zero-pos guard == 0")

-- ---- MAE / MSE / RMSE ----
local tr = from_lua { 3, 5 }
local tt = from_lua { 1, 3 }
local mae = metrics.MAE()
mae:update(tr, tt)
check(near(scalar(mae:compute()), 2.0), "MAE == (2+2)/2 == 2.0")
local mse = metrics.MSE()
mse:update(tr, tt)
check(near(scalar(mse:compute()), 4.0), "MSE == 8/2 == 4.0")
local rmse = metrics.RMSE()
rmse:update(tr, tt)
check(near(scalar(rmse:compute()), 2.0), "RMSE == sqrt(4) == 2.0")
-- multi-batch accumulation
mae:update(from_lua { 1 }, from_lua { 3 })
check(near(scalar(mae:compute()), 2.0), "MAE 2 batches == (4+2)/3 == 2.0")

-- ---- R2 ----
local r2 = metrics.R2()
r2:update(from_lua { 1, 2, 3 }, from_lua { 1, 2, 3 })
check(near(scalar(r2:compute()), 1.0), "R2 perfect == 1.0")
local r2b = metrics.R2()
r2b:update(from_lua { 2, 4, 6 }, from_lua { 1, 2, 3 })
check(near(scalar(r2b:compute()), 0.0), "R2 centered-residual linear == 0.0")

-- ---- MetricCollection ----
local mc = metrics.MetricCollection{
  acc = metrics.Accuracy(),
  mae = metrics.MAE(),
}
mc:update(p1, t1)
local res = mc:compute()
check(near(scalar(res.acc), 1.0), "Collection acc == 1.0")
-- preds=[.8,.2,.9,.1] tgt=[1,0,1,0]: abs diff sum=.6, /4 = .15
check(near(scalar(res.mae), 0.15), "Collection mae (pred vs tgt) == 0.15")
local keys = mc:keys()
check(#keys == 2 and keys[1] == "acc" and keys[2] == "mae",
      "Collection keys ordered")
mc:reset()
local res2 = mc:compute()
-- 0/0 in IEEE-754 is NaN (torchmetrics compute does not guard divide);
-- assert the raw f32 NaN bit pattern so we know reset zeroed the states.
local function is_nan(t)
  local s = t:to_host_bytes()
  local b1, b2, b3, b4 = s:byte(1, 4)
  local bits = b1 + b2 * 256 + b3 * 65536 + b4 * 16777216
  local exp = math.floor(bits / 8388608) % 256
  local mant = bits % 8388608
  return exp == 255 and mant ~= 0
end
check(is_nan(res2.acc), "Collection reset -> acc 0/0 -> NaN")

print(("SMOKE PASS: metrics-lua first batch OK (%d assertions)"
       ):format(ntests))
os.exit(0)