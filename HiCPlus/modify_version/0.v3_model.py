# 加载库
from __future__ import print_function
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
import pandas as pd
import torch
from pytorch_msssim import SSIM
from sklearn.model_selection import train_test_split
import random
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False  


def validate_model(model, val_loader, va_loss):
    model.eval()  # 设置为评估模式
    val_loss = 0.0
    with torch.no_grad():
        for i, (SM0_val, SM1_val, SM2_val) in val_loader:
            _highRes_val = Variable(SM0_val[0]).cuda()
            _lowRes_val = Variable(SM1_val[0]).cuda()
            _signal_val = Variable(SM2_val[0]).cuda()
            _input_SM = (torch.cat((_highRes_val[:,0:1,:,:], _lowRes_val[:,0:1,:,:], _signal_val[:,0:1,:,:]), dim=1)).cuda()
            _input_SA1 = (torch.cat((_highRes_val[:,1:2,:,:], _lowRes_val[:,1:2,:,:], _signal_val[:,1:2,:,:]), dim=1)).cuda()
            _input_SA2 = (torch.cat((_highRes_val[:,2:3,:,:], _lowRes_val[:,2:3,:,:], _signal_val[:,2:3,:,:]), dim=1)).cuda()

            SMC1A_prediction_val, SA1_prediction_val, SA2_prediction_val, SM_enocde_val, SA1_encode_val, SA2_encode_val = model(_input_SM, _input_SA1, _input_SA2)
            val_loss = va_loss(SMC1A_prediction_val ,_input_SM) + va_loss(SA1_prediction_val ,_input_SA1) + va_loss(SA2_prediction_val ,_input_SA2) + np.linalg.norm(
                                                            (SM_enocde_val[:,1:2,:,:] - SA1_encode_val[:,1:2,:,:] - SA2_encode_val[:,1:2,:,:]).detach().cpu().view(-1),2)   

            val_loss += val_loss.item()
    
    val_loss /= len(val_loader)
    return val_loss



binsize = 5000
res = str(int(binsize/1000))
V = 'v7'
# input
factor = 'SMC1A'
inputfile = "/public1/chengrm/cohesin/hicplus/Input"
outmodel = f'/public1/chengrm/cohesin/hicplus/{V}.Output/{factor}/{res}K'
if not os.path.exists(f'/public1/chengrm/cohesin/hicplus/{V}.Output/{factor}'):
    os.makedirs(f'/public1/chengrm/cohesin/hicplus/{V}.Output/{factor}')
if not os.path.exists(outmodel):
    os.makedirs(outmodel)

for i in [21]:
    vv = 'v3'
    chromosome = i
    scalerate = 16

    # 读取数据
    # SMC1A
    SMC1A_highres, size = utils.train_matrix_extract(chromosome, binsize, f'{inputfile}/GM12878_WT_ChIAPET_SMC1A.intra_iPET_ALL.DOWNSAMPLING_43M.hic','SMC1A_hic',0)
    SMC1A_lowres = utils.genDownsample(SMC1A_highres,1/float(scalerate))
    SMC1A_signal, _ = utils.train_matrix_extract(chromosome, binsize, f'{inputfile}/WT_SMC1A.rmOutlier.hic' , 0, size)

    SMC1A_highres_sub, index = utils.train_divide(SMC1A_highres)
    SMC1A_lowres_sub,index = utils.train_divide(SMC1A_lowres)
    SMC1A_signal_sub,index = utils.train_divide(SMC1A_signal)
    print(SMC1A_highres_sub.shape ,SMC1A_lowres_sub.shape ,SMC1A_signal_sub.shape)

    # SA1
    SA1_highres, _ = utils.train_matrix_extract(chromosome, binsize, f'{inputfile}/GM12878_WT_ChIAPET_SA1.intra_iPET_ALL.DOWNSAMPLING_43M.hic', 0, size)
    SA1_lowres = utils.genDownsample(SA1_highres,1/float(scalerate))
    SA1_signal, _ = utils.train_matrix_extract(chromosome, binsize, f'{inputfile}/WT_SA1.rmOutlier.hic', 0, size)

    SA1_highres_sub, index = utils.train_divide(SA1_highres)
    SA1_lowres_sub,index = utils.train_divide(SA1_lowres)
    SA1_signal_sub,index = utils.train_divide(SA1_signal)
    print(SA1_highres_sub.shape ,SA1_lowres_sub.shape ,SA1_signal_sub.shape)

    # SA2
    SA2_highres, _ = utils.train_matrix_extract(chromosome, binsize, f'{inputfile}/GM12878_WT_ChIAPET_SA2.intra_iPET_ALL.DOWNSAMPLING_43M.hic', 0, size)
    SA2_lowres = utils.genDownsample(SA2_highres,1/float(scalerate))
    SA2_signal, _ = utils.train_matrix_extract(chromosome, binsize, f'{inputfile}/WT_SA2.rmOutlier.hic', 0, size)

    SA2_highres_sub, index = utils.train_divide(SA2_highres)
    SA2_lowres_sub,index = utils.train_divide(SA2_lowres)
    SA2_signal_sub,index = utils.train_divide(SA2_signal)
    print(SA2_highres_sub.shape ,SA2_lowres_sub.shape ,SA2_signal_sub.shape)

    use_gpu = 1

    # 卷积核数目及大小
    conv2d1_filters_numbers = 8
    conv2d1_filters_size = 9
    conv2d2_filters_numbers = 8
    conv2d2_filters_size = 1
    conv2d3_filters_numbers = 1
    conv2d3_filters_size = 5

    epochs = 3500
    HiC_max_value = 100
    batch_size = 512
    # batch_size = 800

    SMC1A_highres_sample = np.minimum(HiC_max_value, SMC1A_highres_sub.astype(np.float32))
    SMC1A_lowres_sample = np.minimum(HiC_max_value, SMC1A_lowres_sub.astype(np.float32) * scalerate)
    SMC1A_signal_sample = np.minimum(HiC_max_value, SMC1A_signal_sub.astype(np.float32))

    SA1_highres_sample = np.minimum(HiC_max_value, SA1_highres_sub.astype(np.float32))
    SA1_lowres_sample = np.minimum(HiC_max_value, SA1_lowres_sub.astype(np.float32) * scalerate)
    SA1_signal_sample = np.minimum(HiC_max_value, SA1_signal_sub.astype(np.float32))

    SA2_highres_sample = np.minimum(HiC_max_value, SA2_highres_sub.astype(np.float32))
    SA2_lowres_sample = np.minimum(HiC_max_value, SA2_lowres_sub.astype(np.float32) * scalerate)
    SA2_signal_sample = np.minimum(HiC_max_value, SA2_signal_sub.astype(np.float32))


    # ## 将高质量的 SMC1A、SA1、SA2 Hi-C 样本重新整形为训练的目标值
    # sample_size = SMC1A_highres_sample.shape[-1]
    # padding = conv2d1_filters_size + conv2d2_filters_size + conv2d3_filters_size - 3
    # half_padding = padding // 2
    # output_length = sample_size - padding

    # SMC1A_Y = np.zeros((SMC1A_highres_sample.shape[0], 3, sample_size - padding, sample_size - padding), dtype=np.float32)
    # for i in range(SMC1A_highres_sample.shape[0]):
    #     no_padding_highres_sample = SMC1A_highres_sample[i, 0, half_padding:(sample_size - half_padding), half_padding:(sample_size - half_padding)]
    #     no_padding_lowres_sample = SMC1A_lowres_sample[i, 0, half_padding:(sample_size - half_padding), half_padding:(sample_size - half_padding)]
    #     no_padding_signal_sample = SMC1A_signal_sample[i, 0, half_padding:(sample_size - half_padding), half_padding:(sample_size - half_padding)]
    #     SMC1A_Y[i, 0, :, :] = no_padding_highres_sample
    #     SMC1A_Y[i, 1, :, :] = no_padding_lowres_sample
    #     SMC1A_Y[i, 2, :, :] = no_padding_signal_sample

    # SA1_Y = np.zeros((SA1_highres_sample.shape[0], 3, sample_size - padding, sample_size - padding), dtype=np.float32)
    # for i in range(SA1_highres_sample.shape[0]):
    #     no_padding_highres_sample = SA1_highres_sample[i, 0, half_padding:(sample_size - half_padding), half_padding:(sample_size - half_padding)]
    #     no_padding_lowres_sample = SA1_lowres_sample[i, 0, half_padding:(sample_size - half_padding), half_padding:(sample_size - half_padding)]
    #     no_padding_signal_sample = SA1_signal_sample[i, 0, half_padding:(sample_size - half_padding), half_padding:(sample_size - half_padding)]
    #     SA1_Y[i, 0, :, :] = no_padding_highres_sample
    #     SA1_Y[i, 1, :, :] = no_padding_lowres_sample
    #     SA1_Y[i, 2, :, :] = no_padding_signal_sample

    # SA2_Y = np.zeros((SA2_highres_sample.shape[0], 3, sample_size - padding, sample_size - padding), dtype=np.float32)
    # for i in range(SA2_highres_sample.shape[0]):
    #     no_padding_highres_sample = SA2_highres_sample[i, 0, half_padding:(sample_size - half_padding), half_padding:(sample_size - half_padding)]
    #     no_padding_lowres_sample = SA2_lowres_sample[i, 0, half_padding:(sample_size - half_padding), half_padding:(sample_size - half_padding)]
    #     no_padding_signal_sample = SA2_signal_sample[i, 0, half_padding:(sample_size - half_padding), half_padding:(sample_size - half_padding)]
    #     SA2_Y[i, 0, :, :] = no_padding_highres_sample
    #     SA2_Y[i, 1, :, :] = no_padding_lowres_sample
    #     SA2_Y[i, 2, :, :] = no_padding_signal_sample

    # # 目标矩阵
    # SMC1A_Y = []
    # for i in range(SMC1A_signal_sample.shape[0]):
    #     no_padding_sample = SMC1A_signal_sample[i][0][half_padding:(sample_size-half_padding) , half_padding:(sample_size - half_padding)]
    #     SMC1A_Y.append(no_padding_sample)
    # SMC1A_Y = np.array(SMC1A_Y).astype(np.float32)

    # SA1_Y = []
    # for i in range(SA1_signal_sample.shape[0]):
    #     no_padding_sample = SA1_signal_sample[i][0][half_padding:(sample_size-half_padding) , half_padding:(sample_size - half_padding)]
    #     SA1_Y.append(no_padding_sample)
    # SA1_Y = np.array(SA1_Y).astype(np.float32)

    # SA2_Y = []
    # for i in range(SA2_signal_sample.shape[0]):
    #     no_padding_sample = SA2_signal_sample[i][0][half_padding:(sample_size-half_padding) , half_padding:(sample_size - half_padding)]
    #     SA2_Y.append(no_padding_sample)
    # SA2_Y = np.array(SA2_Y).astype(np.float32)



    # 合并SMC1A、SA1和SA2的数据
    highres_sample = np.concatenate((SMC1A_highres_sample, SA1_highres_sample, SA2_highres_sample), axis=1)
    lowres_sample = np.concatenate((SMC1A_lowres_sample, SA1_lowres_sample, SA2_lowres_sample), axis=1)
    signal_sample = np.concatenate((SMC1A_signal_sample, SA1_signal_sample, SA2_signal_sample), axis=1)
    # Y = np.stack((SMC1A_Y, SA1_Y, SA2_Y), axis=1)

    # 进行训练集和验证集的拆分
    train_highres, val_highres, train_lowres, val_lowres, train_signal, val_signal = \
        train_test_split(highres_sample, lowres_sample, signal_sample, test_size=0.2, random_state=42)

    # 转换为 TensorDataset
    train_highres_set = data.TensorDataset(torch.from_numpy(train_highres), torch.from_numpy(np.zeros(train_highres.shape[0])))
    train_lowres_set = data.TensorDataset(torch.from_numpy(train_lowres), torch.from_numpy(np.zeros(train_lowres.shape[0])))
    train_signal_set = data.TensorDataset(torch.from_numpy(train_signal), torch.from_numpy(np.zeros(train_signal.shape[0])))
    # train_Y_set = data.TensorDataset(torch.from_numpy(train_Y), torch.from_numpy(np.zeros(train_Y.shape[0])))

    val_highres_set = data.TensorDataset(torch.from_numpy(val_highres), torch.from_numpy(np.zeros(val_highres.shape[0])))
    val_lowres_set = data.TensorDataset(torch.from_numpy(val_lowres), torch.from_numpy(np.zeros(val_lowres.shape[0])))
    val_signal_set = data.TensorDataset(torch.from_numpy(val_signal), torch.from_numpy(np.zeros(val_signal.shape[0])))
    # val_Y_set = data.TensorDataset(torch.from_numpy(val_Y), torch.from_numpy(np.zeros(val_Y.shape[0])))


    # 训练集数据加载器
    train_highres_loader = torch.utils.data.DataLoader(train_highres_set, batch_size=batch_size, shuffle=True)
    train_lowres_loader = torch.utils.data.DataLoader(train_lowres_set, batch_size=batch_size, shuffle=True)
    train_signal_loader = torch.utils.data.DataLoader(train_signal_set, batch_size=batch_size, shuffle=True)
    # train_Y_loader = torch.utils.data.DataLoader(train_Y_set, batch_size=batch_size, shuffle=True)

    # 验证集数据加载器
    val_highres_loader = torch.utils.data.DataLoader(val_highres_set, batch_size=batch_size, shuffle=False)
    val_lowres_loader = torch.utils.data.DataLoader(val_lowres_set, batch_size=batch_size, shuffle=False)
    val_signal_loader = torch.utils.data.DataLoader(val_signal_set, batch_size=batch_size, shuffle=False)
    # val_Y_loader = torch.utils.data.DataLoader(val_Y_set, batch_size=batch_size, shuffle=False)

    conv_autoencoder = model.conv_autoencoder2()
    if use_gpu:
        conv_autoencoder = conv_autoencoder.cuda()

    params_to_optimize = [{'params':conv_autoencoder.parameters()}]
    optimizer = optim.SGD(params_to_optimize, lr = 0.01)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.1)
    _loss = nn.MSELoss()
    _va_loss = nn.MSELoss()
    conv_autoencoder.train()

    # 定义初始损失函数
    running_loss = 0.0
    SM_loss = 0.0
    SA1_loss = 0.0
    SA2_loss = 0.0
    signal_loss = 0.0

    save_path = f'{outmodel}/chr{chromosome}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(os.path.join(save_path, f'chr{chromosome}_train_{vv}_loss.txt'), 'w') as log:
        log.write('epoch' + '\t' + 'loss' + '\t' + 'val_loss' +'\t' + 'SMC1A_loss' + '\t' + 'SA1_loss' + '\t' + 'SA2_loss' + '\t' + 'signal_loss' + '\n')
        for epoch in range(epochs):
            loop = tqdm(enumerate(zip(train_highres_loader, train_lowres_loader, train_signal_loader)), total=len(train_highres_loader))
            for i, (SM0, SM1, SM2) in loop:
                _highRes = Variable(SM0[0])
                _lowRes = Variable(SM1[0])
                _signal = Variable(SM2[0])
                # _raw = Variable(RAW[0])

                _input_SM = (torch.cat((_highRes[:,0:1,:,:], _lowRes[:,0:1,:,:], _signal[:,0:1,:,:]), dim=1)).cuda()
                _input_SA1 = (torch.cat((_highRes[:,1:2,:,:], _lowRes[:,1:2,:,:], _signal[:,1:2,:,:]), dim=1)).cuda()
                _input_SA2 = (torch.cat((_highRes[:,2:3,:,:], _lowRes[:,2:3,:,:], _signal[:,2:3,:,:]), dim=1)).cuda()

                SMC1A_prediction, SA1_prediction, SA2_prediction,  SM_encode, SA1_encode, SA2_encode=  conv_autoencoder(_input_SM ,_input_SA1 ,_input_SA2)
                optimizer.zero_grad()
                # _SM_loss = _loss(SMC1A_prediction, _input_SM)
                # _SA1_loss = _loss(SA1_prediction, _input_SA1)
                # _SA2_loss = _loss(SA2_prediction, _input_SA2)
                _SM_loss = _loss(SMC1A_prediction, _input_SM)
                _SA1_loss = _loss(SA1_prediction, _input_SA1)
                _SA2_loss = _loss(SA2_prediction, _input_SA2)
                # _signal_loss = np.linalg.norm((SMC1A_prediction[:,2:3,:,:] - SA1_prediction[:,2:3,:,:] - SA2_prediction[:,2:3,:,:]).detach().cpu().view(-1),2)
                # 第二个encode通道约束signal
                # # 四个箭头的
                # # non_zero_mask = ((_input_SM[:, 2:3, :, :] != 0) + (_input_SA1[:, 2:3, :, :] != 0) + (_input_SA2[:, 2:3, :, :] != 0)) >= 2
                #v3#
                non_zero_mask = (_input_SM[:, 2:3, :, :] != 0) & (_input_SA1[:, 2:3, :, :] != 0) & (_input_SA2[:, 2:3, :, :] != 0)
                #v4#
                # non_zero_mask = (_input_SM[:, 0:1, :, :] != 0) & (_input_SA1[:, 0:1, :, :] != 0) & (_input_SA2[:, 0:1, :, :] != 0)
                
                # _signal_loss = torch.norm((SM_encode[:, 1:2, :, :] - SA1_encode[:, 1:2, :, :] - SA2_encode[:, 1:2, :, :]).masked_select(non_zero_mask).view(-1), 2)
                #v3# 
                _signal_loss = torch.norm((SMC1A_prediction[:,2:3,:,:] - SA1_prediction[:,2:3,:,:] - SA2_prediction[:,2:3,:,:]).masked_select(non_zero_mask).view(-1), 2)
                #v4#
                # _signal_loss = torch.norm((SMC1A_prediction[:,0:1,:,:] - SA1_prediction[:,0:1,:,:] - SA2_prediction[:,0:1,:,:]).masked_select(non_zero_mask).view(-1), 2)
                
                loss = _SM_loss + _SA1_loss + _SA2_loss + _signal_loss   
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                SM_loss += (_SM_loss.item())/(len(train_highres_loader)-1)
                SA1_loss += (_SA1_loss.item())/(len(train_highres_loader)-1)
                SA2_loss += (_SA2_loss.item())/(len(train_highres_loader)-1)
                signal_loss += (_signal_loss.item())/(len(train_highres_loader)-1)
                
                loop.set_description(f'Epoch [{epoch + 1}/{epochs}]')
                loop.set_postfix(loss=running_loss / (len(train_highres_loader) - 1))
            
            scheduler.step()
            # 计算验证集损失
            val_zip = tqdm(enumerate(zip(val_highres_loader, val_lowres_loader, val_signal_loader)), total=len(val_highres_loader))
            val_loss = validate_model(conv_autoencoder, val_loader=val_zip, va_loss = _va_loss)
            # 输出训练和验证损失

            print(f'Epoch [{epoch + 1}/{epochs}], Training Loss: {running_loss / (len(train_highres_loader) - 1)}, Validation Loss: {val_loss}')

            log.write(str(epoch) + '\t' + str(running_loss/(len(train_highres_loader)-1)) + '\t' + f'{val_loss}' + '\t' + f'{SM_loss}' + '\t' + f'{SA1_loss}' + '\t' + f'{SA2_loss}' + '\t' + f'{signal_loss}' + '\n')
            
            running_loss = 0.0  #
            SM_loss = 0.0
            SA1_loss = 0.0
            SA2_loss = 0.0
            signal_loss = 0.0

            if ((epoch + 1) % 50 == 0):
                torch.save(conv_autoencoder.state_dict(), os.path.join(save_path, 'chr' + str(chromosome) + f'_{vv}_' + str(epoch + 1) + str('.model')))
            pass

        log.close() 