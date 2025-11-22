# 加载库
from __future__ import print_function
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import utils
import model
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from math import log10
from torch.utils import data
from time import gmtime, strftime
from torch.autograd import Variable
from joblib import Parallel, delayed
# import sys
# import gzip
# import torch
# import pickle
# import argparse
# import trainConvNet
# import torch.nn as nn
# import torch.optim as optim
# import matplotlib.pyplot as plt
# from torch.autograd import Variable
# from torch.utils.data import DataLoader

# 数据处理
# 染色体长度列表，1-22号染色体
# chrs_length = [249250621, 243199373, 198022430, 191154276, 180915260, 171115067, 159138663, 146364022, 141213431, 135534747, 135006516, 133851895, 115169878, 107349540, 102531392, 90354753, 81195210, 78077248, 59128983, 63025520, 48129895, 51304566]
def train(binsize, res, factor, inputfile, outmodel, chr):
    chromosome = chr
    scalerate = 16
    highres = utils.train_matrix_extract(chromosome, binsize, inputfile)
    print('dividing, filtering and downsampling files...')
    highres_sub, index = utils.train_divide(highres)
    print(highres_sub.shape)
    lowres = utils.genDownsample(highres,1/float(scalerate))
    lowres_sub,index = utils.train_divide(lowres)
    print(lowres_sub.shape)
    # 训练参数设置
    # 设置使用的GPU
    use_gpu = 1
    # 卷积核数目及大小
    conv2d1_filters_numbers = 8
    conv2d1_filters_size = 9
    conv2d2_filters_numbers = 8
    conv2d2_filters_size = 1
    conv2d3_filters_numbers = 1
    conv2d3_filters_size = 5
    # 训练参数
    down_sample_ratio = 16
    epochs = 10
    HiC_max_value = 100
    batch_size = 512
    # 加载训练数据
    low_resolution_samples = lowres_sub.astype(np.float32) * down_sample_ratio
    high_resolution_samples = highres_sub.astype(np.float32)
    low_resolution_samples = np.minimum(HiC_max_value, low_resolution_samples)
    high_resolution_samples = np.minimum(HiC_max_value, high_resolution_samples)
    # Reshape the high-quality Hi-C sample as the target value of the training.
    sample_size = low_resolution_samples.shape[-1]
    padding = conv2d1_filters_size + conv2d2_filters_size + conv2d3_filters_size - 3
    half_padding = padding // 2
    output_length = sample_size - padding
    # Y是预测得到的高分辨率矩阵
    Y = []
    for i in range(high_resolution_samples.shape[0]):
        no_padding_sample = high_resolution_samples[i][0][half_padding:(sample_size-half_padding) , half_padding:(sample_size - half_padding)]
        Y.append(no_padding_sample)
    Y = np.array(Y).astype(np.float32)
    print(low_resolution_samples.shape, Y.shape)
    ## 数据加载器
    lowres_set = data.TensorDataset(torch.from_numpy(low_resolution_samples), torch.from_numpy(np.zeros(low_resolution_samples.shape[0])))
    lowres_loader = torch.utils.data.DataLoader(lowres_set, batch_size=batch_size, shuffle=False)
    hires_set = data.TensorDataset(torch.from_numpy(Y), torch.from_numpy(np.zeros(Y.shape[0])))
    hires_loader = torch.utils.data.DataLoader(hires_set, batch_size=batch_size, shuffle=False)
    ## 卷积神经网络实例化
    Net = model.Net(40, 28)
    # 是否使用GPU
    if use_gpu:
        Net = Net.cuda()
    # 使用梯度下降优化
    optimizer = optim.SGD(Net.parameters(), lr = 0.00001)
    _loss = nn.MSELoss()
    Net.train()
    # 定义初始损失函数
    running_loss = 0.0
    running_loss_validate = 0.0
    reg_loss = 0.0
    # write the log file to record the training process
    save_path = f'{outmodel}/chr{chromosome}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(os.path.join(save_path, f'chr{chromosome}_train.txt'), 'w') as log:
        for epoch in range(0, 3500):
            # 存储epoch和Loss
            epochs = []
            losses = []
            loop = tqdm(enumerate(zip(lowres_loader, hires_loader)), total = len(lowres_loader))
            for i, (v1, v2) in loop:
                if (i == len(lowres_loader) - 1):
                    continue
                # print(i)
                _lowRes, _ = v1
                _highRes, _ = v2
                _lowRes = Variable(_lowRes)
                _highRes = Variable(_highRes).unsqueeze(1)
                if use_gpu:
                    _lowRes = _lowRes.cuda()
                    _highRes = _highRes.cuda()
                optimizer.zero_grad()
                y_prediction = Net(_lowRes)

                loss = _loss(y_prediction, _highRes)
                loss.backward()
                optimizer.step()
                # 计算每个epoch的损失值的和
                running_loss += loss.item()
                # 更新进度条
                loop.set_description(f'Epoch [{epoch + 1}/{3500}]')
                loop.set_postfix(loss = running_loss/(len(lowres_loader)-1))
            print('-------', i, epoch, running_loss/i, strftime("%Y-%m-%d %H:%M:%S", gmtime()))
            ## 添加元素
            epochs.append(epoch + 1)
            losses.append(running_loss)   
            log.write(str(epoch) + ', ' + str(running_loss) +', '+ strftime("%Y-%m-%d %H:%M:%S", gmtime())+ '\n')
            running_loss = 0.0
            running_loss_validate = 0.0
            # save the model every 100 epoches
            if (epoch % 100 == 0):
                torch.save(Net.state_dict(), os.path.join(save_path, 'chr' + str(chromosome) + "_" + str(epoch) + str('.model')))
            pass
    log.close()


binsize = 5000
res = str(int(binsize/1000))
factor = 'SMC1A'
inputfile = "/public1/xinyu/CohesinProject/DeepLearning_Cohesin/0.HiCPlus/Input/GM12878_WT_ChIAPET_SMC1A.intra_iPET_ALL.hic"
outmodel = f'/public1/xinyu/CohesinProject/DeepLearning_Cohesin/0.HiCPlus/output/{factor}/1D'
# chrN = 21
# scale = 16
## 对染色体进行循环，逐条进行训练
chr_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 'X', 'Y']
for i in [21]:
    train(binsize = binsize, 
          res = res,
          factor = factor,
          inputfile = inputfile,
          outmodel = outmodel,
          chr = i)
