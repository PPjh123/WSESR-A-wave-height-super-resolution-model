import numpy as np
from skimage.metrics import structural_similarity as ssim
import math

def MAE(pred, true):
    return np.mean(np.abs(pred - true))

def MSE(pred, true):
    return np.mean((pred - true) ** 2)

def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

def PSNR(pred, true):
    # 将图像转换为浮动类型以避免数据溢出
    original = true.astype(np.float64)
    generated = pred.astype(np.float64)

    # 计算MSE（均方误差）
    mse = np.mean((original - generated) ** 2)
    if mse == 0:
        return float('inf')  # 如果MSE为0，PSNR为无穷大
    # 获取最大像素值
    PIXEL_MAX = np.max(original)
    # 计算PSNR
    psnr_value = 10 * math.log10((PIXEL_MAX ** 2) / mse)
    return psnr_value

def SSIM(original, generated):
    ssim_value, _ = ssim(original, generated, full=True, data_range=original.max() - original.min())
    return ssim_value

def metric(pred, true):
    mae = MAE(pred, true)
    rmse = RMSE(pred, true)
    psnr = PSNR(pred, true)
    ssim = SSIM(pred, true)
    return mae, rmse, psnr, ssim
