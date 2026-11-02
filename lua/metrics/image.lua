-- lua/metrics/image.lua -- 图像指标(首批:PSNR)
-- PSNR = 20·log10(MAX) - 10·log10(MSE),MAX = data_range(默认 1.0)。
-- torchmetrics.image.PeakSignalNoiseRatio 语义;MSE 用增量累计(sumsq,count)。

local class = require "pl.class"
local paddle = require "paddle"
local Metric = require "metrics.base"

local peak_signal_noise_ratio = class(Metric)

function peak_signal_noise_ratio:_init(config)
  Metric._init(self, config)
  self.data_range = (config and config.data_range) or 1.0
  self:add_state{ "sum_sq", paddle.zeros{ 1 }, dist_reduce_fx = "sum" }
  self:add_state{ "count", paddle.zeros{ 1 }, dist_reduce_fx = "sum" }
end

function peak_signal_noise_ratio:update(preds, target)
  self:_check_tensors(preds, target)
  local diff = paddle.subtract{ preds, target }
  self.sum_sq = self.sum_sq + paddle.sum{ diff * diff }
  self.count = self.count + self:_numel_as_tensor(preds)
end

function peak_signal_noise_ratio:compute()
  local mse = paddle.divide{ self.sum_sq, self.count }
  local mse_floor = paddle.maximum{ mse, paddle.full{ { 1 }, 1e-10 } }
  -- 20·log10(MAX) - 10·log10(MSE);标量乘用 full 张量(元方法只收双 tensor)
  local maxdb = paddle.full{ { 1 }, 20 } * paddle.log10{
    paddle.full{ { 1 }, self.data_range } }
  local msedb = paddle.full{ { 1 }, 10 } * paddle.log10{ mse_floor }
  return paddle.subtract{ maxdb, msedb }
end

return {
  PSNR = peak_signal_noise_ratio,
}