import numpy as np
import math
import os
from PIL import Image
import matplotlib.pyplot as plt
from metrics import metric

dataset_name='S4_2021_2024'
# data loading
feature = np.load('D:\DATA/Wave_height_super_resolution/Data/ERA5-3h/' + dataset_name + '.npy')
label = np.load('D:\DATA/Wave_height_super_resolution/Data/Copernicus-3h/' + dataset_name + '.npy')
test_len = math.ceil(feature.shape[2] * 0.2)
LR=feature[:,:,-test_len:]
HR=label[:,:,-test_len:]

SR=np.empty_like(HR)
for i in range(LR.shape[2]):
    # 使用双三次插值将矩阵放大到 6 倍大小
    # 将 NumPy 数组转换为 PIL 图像对象
    input_image = Image.fromarray(LR[:, :, i])
    # 对图像进行 6 倍上采样，使用双三次插值
    output_image = input_image.resize((LR.shape[1] * 6, LR.shape[0] * 6), Image.BICUBIC)
    # 转换回 NumPy 数组
    SR[:, :, i] = np.array(output_image)


# result save
folder_path = 'D:\DATA/Wave_height_super_resolution/result/Bicubic/' + dataset_name + '/'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
mae, rmse, psnr, ssim = metric(SR, HR)
print('mae:{:.5f},rmse:{:.5f},psnr:{:.5f},ssim:{:.5f}'.format(mae, rmse, psnr, ssim))
f = open(folder_path + "metrics.txt", 'a')
f.write(dataset_name + "  \n")
f.write(' mae:{:.5f}'.format(mae))
f.write('\n')
f.write(' rmse:{:.5f}'.format(rmse))
f.write('\n')
f.write(' psnr:{:.5f}'.format(psnr))
f.write('\n')
f.write(' ssim:{:.5f}'.format(ssim))
f.write('\n')
f.close()
np.save(folder_path + 'LR.npy', np.transpose(LR, (2,0,1)))
np.save(folder_path + 'SR.npy', np.transpose(SR, (2,0,1)))
np.save(folder_path + 'HR.npy', np.transpose(HR, (2,0,1)))

