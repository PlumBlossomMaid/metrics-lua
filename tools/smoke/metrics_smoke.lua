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
local function from_lua(vals, shape)
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
  return core.from_host_bytes(table.concat(bytes), shape or { #vals },
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

-- ---- multiclass Accuracy (argmax path) ----
-- logits [4,3]: argmax -> [1,2,3,1] (1-based);target [1,2,3,2]: 3/4 hit
local acc3 = metrics.Accuracy{ multiclass = true, num_classes = 3 }
acc3:update(from_lua({ 3, 1, 0, 0, 2, 1, 1, 0, 4, 2, 0, 1 }, { 4, 3 }),
            from_lua { 1, 2, 3, 2 })
check(near(scalar(acc3:compute()), 0.75), "multiclass Accuracy == 3/4")

-- ---- ConfusionMatrix ----
local cm = metrics.ConfusionMatrix{ num_classes = 3 }
cm:update(from_lua { 1, 2, 2, 3 }, from_lua { 1, 2, 1, 1 })
local m = cm:compute()  -- [3,3]
-- expected: CM[1,1]=1 (pred1,tgt1), CM[2,1]=1, CM[2,2]=1, CM[3,1]=1
local bytes = m:to_host_bytes()
check(near(f32_at(bytes, 1), 1) and near(f32_at(bytes, 17), 1),
      "ConfusionMatrix diag-ish counts (0,0)/(2,0)")
-- CM[1,2]=0: byte 5
check(near(f32_at(bytes, 5), 0), "ConfusionMatrix zero off-diag")
cm:reset()
cm:update(from_lua { 3 }, from_lua { 3 })  -- CM[3,3] += 1
local m2 = cm:compute()
-- [3,3] row-major: element (3,3) is 0-based 8 -> byte 8*4+1 = 33
check(near(f32_at(m2:to_host_bytes(), 33), 1), "ConfusionMatrix reset+CM[3,3]")

-- ---- PearsonCorrCoef ----
local pc = metrics.PearsonCorrCoef()
pc:update(from_lua { 1, 2, 3, 4 }, from_lua { 2, 4, 6, 8 })
check(near(scalar(pc:compute()), 1.0), "Pearson perfect linear == 1.0")

-- ---- MeanMetric / SumMetric ----
local mm = metrics.MeanMetric()
mm:update(from_lua { 1, 3 })   -- sum=4, count=2
mm:update(5)                    -- +5, +1
check(near(scalar(mm:compute()), 3.0), "MeanMetric == (1+3+5)/3 == 3")
local sm = metrics.SumMetric()
sm:update(from_lua { 1, 2 })
sm:update(3)
check(near(scalar(sm:compute()), 6.0), "SumMetric == 1+2+3 == 6")

-- ---- CatMetric ----
-- 收集 3 个 [2] 张量,沿 axis=1 竖排拼接 -> [6]
local cat = metrics.CatMetric()
cat:update(from_lua { 1, 2 })
cat:update(from_lua { 3, 4 })
cat:update(from_lua { 5, 6 })
local cbs = cat:compute():to_host_bytes()
check(f32_at(cbs, 1) == 1 and f32_at(cbs, 5) == 2 and f32_at(cbs, 9) == 3
      and f32_at(cbs, 13) == 4 and f32_at(cbs, 17) == 5 and f32_at(cbs, 21) == 6,
      "CatMetric concat 3x[2] -> [1..6]")
-- reset 后收集态清空
cat:reset()
cat:update(from_lua { 9, 8 })
local c2b = cat:compute():to_host_bytes()
check(f32_at(c2b, 1) == 9 and f32_at(c2b, 5) == 8, "CatMetric reset + reuse")

-- ---- PSNR ----
-- preds=[0,1] target=[1,1]: MSE=0.5,
-- PSNR = 20*log10(1) - 10*log10(0.5) = 0 + 3.0103
local psnr = metrics.PSNR{ data_range = 1.0 }
psnr:update(from_lua { 0, 1 }, from_lua { 1, 1 })
check(near(scalar(psnr:compute()), 3.0103), "PSNR MSE=0.5 == 3.0103")

-- ---- retrieval ----
-- AUROC: score=[.3,.8,.6,.9] target=[0,1,0,1]
-- asc ranks: .3->1 .6->2 .8->3 .9->4;pos ranks 3+4=7
-- AUC = (7 - 2*3/2)/(2*2) = (7-3)/4 = 1.0
local auroc = metrics.AUROC()
auroc:update(from_lua { 0.3, 0.8, 0.6, 0.9 }, from_lua { 0, 1, 0, 1 })
check(near(scalar(auroc:compute()), 1.0), "AUROC perfect ordering == 1.0")
-- reverse: score=[.9,.8,.6,.3] target=[1,0,1,0]
-- pos ranks: pos@1(.9)->4, pos@3(.6)->2 -> sum 6 -> (6-3)/4 = .75
local auroc2 = metrics.AUROC()
auroc2:update(from_lua { 0.9, 0.8, 0.6, 0.3 }, from_lua { 1, 0, 1, 0 })
check(near(scalar(auroc2:compute()), 0.75), "AUROC partial ordering == 0.75")

-- AveragePrecision: same as first AUC case: desc order 4(1),2(1),3,1
-- k=1 tp=1 ap+=1; k=2 tp=2 ap+=1 => ap=2; /n_pos(2) = 1.0
local ap = metrics.AveragePrecision()
ap:update(from_lua { 0.3, 0.8, 0.6, 0.9 }, from_lua { 0, 1, 0, 1 })
check(near(scalar(ap:compute()), 1.0), "AveragePrecision perfect == 1.0")

-- MRR: first relevant at rank 1 (score .9) -> 1.0
local mrr = metrics.MRR()
mrr:update(from_lua { 0.3, 0.8, 0.6, 0.9 }, from_lua { 0, 1, 0, 1 })
check(near(scalar(mrr:compute()), 1.0), "MRR rank1 == 1.0")
-- MRR: no relevant at rank1 -> 0.5
local mrr2 = metrics.MRR()
mrr2:update(from_lua { 0.9, 0.8 }, from_lua { 0, 1 })
check(near(scalar(mrr2:compute()), 0.5), "MRR rank2 == 0.5")

-- NDCG: rel=[0,1,0,1] both orders give DCG=1/log2(2)+1/log2(3) -> 1.0
local ndcg = metrics.NDCG()
ndcg:update(from_lua { 0.3, 0.8, 0.6, 0.9 }, from_lua { 0, 1, 0, 1 })
check(near(scalar(ndcg:compute()), 1.0), "NDCG perfect ranking == 1.0")
-- NDCG: reversed-relevance case
-- score desc order -> positions 1,2,3,4 with rel [0,1,0,1]
-- DCG = 1/log2(3) + 1/log2(5) = 0.63092975357 + 0.43067655807 = 1.06160631164
-- IDCG(target [1,1,0,0] desc) = 1/log2(2) + 1/log2(3) = 1.63092975357
-- NDCG = 1.06160631164 / 1.63092975357 = 0.65092092981
local ndcg2 = metrics.NDCG()
ndcg2:update(from_lua { 0.9, 0.8, 0.6, 0.3 }, from_lua { 0, 1, 0, 1 })
check(near(scalar(ndcg2:compute()), 0.6509209), "NDCG partial == 0.6509209")

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