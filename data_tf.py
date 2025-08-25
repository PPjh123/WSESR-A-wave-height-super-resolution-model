import h5py
import numpy as np
from scipy.io import loadmat
path='D:\DATA/Wave_height_super_resolution/Data/ERA5-3h-wt/'
dataset=['S1_2021_2024','S2_2021_2024','S3_2021_2024']
# 加载 .mat 文件
for data in dataset:
    with h5py.File(path+data+'.mat') as f:  # 替换为你的文件名
        # 查看 .mat 文件中的变量
        print(list(f.keys()))
        # 假设我们想转换其中的变量 'data'（变量名称根据 .mat 文件内容来定）
        wavedata = f['swh_wt'][:]
        #wavedata = f['swh'][:] # 'data' 是你要提取的变量名称
        # 将数据保存为 .npy 文件
        transposed_data = np.transpose(wavedata, (3, 2, 1, 0))
        #transposed_data = np.transpose(wavedata, (2, 1, 0))
        np.save(path + data + '.npy',transposed_data)  # 替换为你想要保存的文件名



