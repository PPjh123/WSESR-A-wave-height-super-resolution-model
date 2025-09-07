import numpy as np
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
from torch import optim
import os
import math
import torch
import torch.nn as nn
from tools import EarlyStopping
from metrics import metric
import argparse
import random
import torch.nn.functional as F


def set_seed(seed):
    random.seed(seed)  # Python随机种子
    np.random.seed(seed)  # Numpy随机种子
    torch.manual_seed(seed)  # CPU随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # GPU随机种子（单个GPU）
        torch.cuda.manual_seed_all(seed)  # GPU随机种子（所有GPU）

set_seed(2024)

class DSC_bolck(nn.Module):
    def __init__(self, input_dim,output_dim, kernel_size, padding ):
        super().__init__()
        self.dwconv = nn.Conv2d(input_dim, input_dim, kernel_size=kernel_size, padding=padding, groups=input_dim) # depthwise conv
        #self.norm = LayerNorm(input_dim, eps=1e-6)
        self.pwconv1 = nn.Linear(input_dim, 4* input_dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4* input_dim, output_dim)

    def forward(self, x):
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
        return x

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
class UpsampleBLock(nn.Module):
    def __init__(self, in_channels):
        super(UpsampleBLock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels * 2 ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle2 = nn.PixelShuffle(2)
        self.con2 = nn.Conv2d(in_channels, in_channels * 3 ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle3 = nn.PixelShuffle(3)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pixel_shuffle2(x)
        x = self.prelu(x)
        x = self.con2(x)
        x = self.pixel_shuffle3(x)
        x = self.prelu(x)
        return x

class DSCRA(nn.Module):
    def __init__(self, args):
        super(DSCRA, self).__init__()
        self.res_scale = args.res_scale
        self.input_conv = DSC_bolck(args.in_channels, args.hidden_channels, kernel_size=7, padding=3)

        self.Res = nn.Sequential(DSC_bolck(args.hidden_channels,args.hidden_channels, kernel_size=3, padding=1),
                                nn.ReLU(inplace=True),
                                DSC_bolck(args.hidden_channels, args.hidden_channels, kernel_size=3, padding=1)
                                )
        self.SA_conv = nn.Conv2d(1, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

        self.feature_reconstruct = DSC_bolck(args.hidden_channels, args.hidden_channels, kernel_size=3, padding=1)

        self.upsample = UpsampleBLock(args.hidden_channels)

        self.output_conv = DSC_bolck(args.hidden_channels, 1, kernel_size=3, padding=1)

    def forward(self, x, args):
        x = x.permute(0,3,1,2)
        x = self.input_conv(x)
        residual = x
        x = self.Res(x)
        #avg_out = torch.mean(x, dim=1, keepdim=True)  # 平均池化，结果形状为 (B, 1, H, W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 最大池化，结果形状为 (B, 1, H, W)
        #SA = torch.cat([avg_out, max_out], dim=1)  # 拼接后形状为 (B, 2, H, W)
        SA = self.SA_conv(max_out)  # 卷积，结果形状为 (B, 1, H, W)
        SA = self.sigmoid(SA)  # Sigmoid 激活
        x = x * SA
        x = self.feature_reconstruct(x)
        x += self.res_scale * residual  # Long skip connection
        x = self.upsample(x)
        x = self.output_conv(x)
        x = x.squeeze(1)
        return x


class My_dataset(Dataset):
    def __init__(self, feature, label):
        self.feature = feature
        self.label = label
        self.data_len = label.shape[2]

    def __getitem__(self, idx):
        data_feature = self.feature[:,:,:, idx]
        data_label = self.label[:,:,idx]
        return data_feature, data_label

    def __len__(self):
        iter_len = self.data_len
        return iter_len

##### valid ######
def valid(valid_loader):
    total_loss = []
    model.eval()
    criterion = nn.L1Loss()
    with torch.no_grad():
        for i, (data_feature, data_label) in enumerate(valid_loader):
            data_feature = data_feature.to(dtype=torch.float32, device='cuda')
            data_label = data_label.to(dtype=torch.float32, device='cuda')
            outputs = model(data_feature,args)
            loss = criterion(outputs, data_label)
            total_loss.append(loss.item())

        avg_loss = np.average(total_loss)

    return avg_loss


##### train #######
def train(train_loader, valid_loader, args):
    path = os.path.join('./checkpoints/'+args.model_name,dataset_name)
    if not os.path.exists(path):
        os.makedirs(path)
    scaler = torch.amp.GradScaler()#使用混合精度，减少显存
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    model_optim = optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.L1Loss()
    for epoch in range(args.epoch):
        train_loss = []
        model.train()

        for i, (data_feature, data_label) in enumerate(train_loader):
            model_optim.zero_grad()
            data_feature = data_feature.to(dtype=torch.float32, device='cuda')
            data_label = data_label.to(dtype=torch.float32, device='cuda')

            with torch.amp.autocast('cuda'):
                outputs = model(data_feature,args)
                loss = criterion(outputs, data_label)

            scaler.scale(loss).backward()
            scaler.step(model_optim)
            scaler.update()
            train_loss.append(loss.item())

        StepLR(model_optim, step_size=args.step_size, gamma=args.gamma)
        valid_loss = valid(valid_loader)
        early_stopping(valid_loss, model, path)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    best_model_path = path + '/' + 'checkpoint.pth'
    model.load_state_dict(torch.load(best_model_path,weights_only=True))
    return

def test(test_loader,args):
    print('loading model')
    model.load_state_dict(torch.load(os.path.join(
         './checkpoints/' + args.model_name+'/'+dataset_name, 'checkpoint.pth'),weights_only=True))
    preds = torch.tensor([])
    trues = torch.tensor([])

    model.eval()
    with torch.no_grad():
        for i, (data_feature, data_label) in enumerate(test_loader):
            data_feature = data_feature.to(dtype=torch.float32, device='cuda')
            data_label = data_label.to(dtype=torch.float32, device='cuda')
            outputs = model(data_feature, args)
            pred = outputs.detach().cpu() *(max -min) + min
            true = data_label.detach().cpu() *(max -min) + min

            preds=torch.cat([preds, pred],0)
            trues=torch.cat([trues, true],0)

        preds = preds.numpy()
        trues = trues.numpy()
        print('trues_len is:{}'.format(len(trues)))
        print('preds_len is:{}'.format(len(preds)))

    return trues, preds


def read_data(feature,label,args):
    data_len = label.shape[2]
    train_border = math.floor(data_len * 0.7)
    valid_border = data_len - test_len

    train_label = label[:,:,0:train_border]
    # normalize
    data_min = np.min(train_label)
    data_max = np.max(train_label)

    train_feature_norm= (feature[:,:,:,0:train_border] - data_min) / (data_max - data_min)
    valid_feature_norm = (feature[:,:,:,train_border:valid_border] - data_min) / (data_max - data_min)
    test_feature_norm = (feature[:,:,:,-test_len:] - data_min) / (data_max - data_min)

    train_label_norm = (label[:,:,0:train_border] - data_min) / (data_max - data_min)
    valid_label_norm = (label[:,:,train_border:valid_border] - data_min) / (data_max - data_min)
    test_label_norm = (label[:,:,-test_len:] - data_min) / (data_max - data_min)

    train_set = My_dataset(train_feature_norm,train_label_norm)
    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True, num_workers=0,
                              drop_last=True)
    valid_set = My_dataset(valid_feature_norm,valid_label_norm)
    valid_loader = DataLoader(dataset=valid_set, batch_size=args.batch_size, shuffle=True, num_workers=0,
                              drop_last=True)
    test_set = My_dataset(test_feature_norm,test_label_norm)
    test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
    return train_loader, valid_loader, test_loader, data_max, data_min
########################################################################################################################

parser = argparse.ArgumentParser(description='Parameters for boundary extension')
# basic config
parser.add_argument('--model_name', type=str, default='WT_DSCR_SA')
# conv
parser.add_argument('--in_channels', type=int, default=5)
parser.add_argument('--hidden_channels', type=int, default=64)
parser.add_argument('--res_scale', type=float, default=1)
# sub-pixel for upsampling
parser.add_argument('--scale_factor', type=int, default=6)

# training set
parser.add_argument('--epoch', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--learning_rate', type=float, default=0.0005, help='optimizer learning rate')
parser.add_argument('--step_size', type=int, default=50,
                    help='step size (epochs) of learning rate adjusting')
parser.add_argument('--gamma', type=float, default=0.1, help='drop factor of learning rate adjusting')
parser.add_argument('--patience', type=int, default=30, help='patience of early stopping')
args = parser.parse_args()

############ s1 ##################
dataset_name='S1_2021_2024'
wt_feature = np.load('D:\DATA/Wave_height_super_resolution/Data/ERA5-3h-wt/' + dataset_name + '.npy')
wt_feature = wt_feature[:, :, :4, :]# height, width, feature, length
label = np.load('D:\DATA/Wave_height_super_resolution/Data/Copernicus-3h/' + dataset_name + '.npy')# height, width, length
lr_feature = np.load('D:\DATA/Wave_height_super_resolution/Data/ERA5-3h/' + dataset_name + '.npy')# height, width, length
test_len = math.ceil(label.shape[2] * 0.2)
temp = lr_feature.reshape(lr_feature.shape[0], lr_feature.shape[1], 1, lr_feature.shape[2])# height, width, feature, length
feature = np.concatenate((temp, wt_feature), 2)# height, width, feature, length

#train model
model = DSCRA(args).cuda()
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")
train_loader, valid_loader, test_loader, max, min = read_data(feature, label, args)
train(train_loader, valid_loader, args)
true, pred = test(test_loader, args)  #########
torch.cuda.empty_cache() # 释放GPU内存

# result save
folder_path = 'D:\DATA/Wave_height_super_resolution/result/' + args.model_name + '/' + dataset_name + '/'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
mae, rmse, psnr, ssim = metric(pred, true)
print('mae:{:.5f},rmse:{:.5f},mape:{:.5f},r:{:.5f}'.format(mae, rmse, psnr, ssim))
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
np.save(folder_path + 'SR.npy', pred)

