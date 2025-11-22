#%% 加载库
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import model
import utils
import straw
import torch
import numpy as np
import torch.nn as nn
from scipy import sparse
import torch.optim as optim
from torch.utils import data
from datetime import datetime
from time import gmtime, strftime
from torch.autograd import Variable
from scipy.sparse import csr_matrix, coo_matrix, vstack, hstack

#%% 定义函数
def predict(SM, SA1, SA2, inmodel):


    prediction_hic = np.zeros((SM.shape[0], SM.shape[0], 3))
    prediction_signal = np.zeros((SM.shape[0], SM.shape[0], 3))


    for (SM_sample, index_0) ,(SA1_sample, index_1) ,(SA2_sample, index_2)in zip(utils.divide(SM),utils.divide(SA1),utils.divide(SA2)):

        batch_size = SM_sample.shape[0] #256

        lowres_set_0 = data.TensorDataset(torch.from_numpy(SM_sample), torch.from_numpy(np.zeros(SM_sample.shape[0])))
        lowres_set_1 = data.TensorDataset(torch.from_numpy(SA1_sample), torch.from_numpy(np.zeros(SA1_sample.shape[0])))
        lowres_set_2 = data.TensorDataset(torch.from_numpy(SA2_sample), torch.from_numpy(np.zeros(SA2_sample.shape[0])))
        try:
            lowres_loader_0 = torch.utils.data.DataLoader(lowres_set_0, batch_size = batch_size, shuffle = False)
            lowres_loader_1 = torch.utils.data.DataLoader(lowres_set_1, batch_size = batch_size, shuffle = False)
            lowres_loader_2 = torch.utils.data.DataLoader(lowres_set_2, batch_size = batch_size, shuffle = False)
        except:
            continue

        m = model.conv_autoencoder_pro()

        m.load_state_dict(torch.load(inmodel, map_location = torch.device('cpu')))
        if torch.cuda.is_available():
            m = m.cuda()
            use_gpu = True
        else:
            use_gpu = False

        # 输入维度是3，还得更改
        loop = enumerate(zip(lowres_loader_0, lowres_loader_1, lowres_loader_2))
        for i, (v0, v1, v2) in loop:
            SM_input = v0[0].permute(0, 4, 1, 2, 3).squeeze(dim = 2)
            SA1_input = v1[0].permute(0, 4, 1, 2, 3).squeeze(dim = 2)
            SA2_input = v2[0].permute(0, 4, 1, 2, 3).squeeze(dim = 2)
            SM_input = Variable(SM_input).float()
            SA1_input = Variable(SA1_input).float()
            SA2_input = Variable(SA2_input).float()
    
            if use_gpu:
                SM_input = SM_input.cuda()
                SA1_input = SA1_input.cuda()
                SA2_input = SA2_input.cuda()
            
            SM_prediction, SA1_prediction, SA2_prediction, SM_encode, SA1_encode, SA2_encode = m(SM_input, SA1_input, SA2_input)

        SM_prediction = SM_prediction.data.cpu().numpy()
        SA1_prediction = SA1_prediction.data.cpu().numpy()
        SA2_prediction = SA2_prediction.data.cpu().numpy()
        # SM_encode = SM_encode.data.cpu().numpy()
        # SA1_encode = SA1_encode.data.cpu().numpy()
        # SA2_encode = SA2_encode.data.cpu().numpy()


        y_predict_hic = SM_prediction[:,0:1,:,:]
        y_predict_signal = SM_prediction[:,2:3,:,:]
        length = int(y_predict_hic.shape[2])
        y_predict_hic = np.reshape(y_predict_hic, (y_predict_hic.shape[0], length, length))
        for i in range(0, y_predict_hic.shape[0]):
            x = int(index_0[i][1])
            y = int(index_0[i][2])
            prediction_hic[x:x+40, y:y+40, 0] = y_predict_hic[i]
            # prediction_hic[x+6:x+46, y+6:y+46, 0] = y_predict_hic[i][:, :(prediction_hic[x+6:x+46, y+6:y+46, 0].shape[1])]

        length = int(y_predict_signal.shape[2])
        y_predict_signal = np.reshape(y_predict_signal, (y_predict_signal.shape[0], length, length))
        for i in range(0, y_predict_signal.shape[0]):
            x = int(index_0[i][1])
            y = int(index_0[i][2])
            prediction_signal[x:x+40, y:y+40, 0] = y_predict_signal[i]
            # prediction_signal[x+6:x+46, y+6:y+46, 0] = y_predict_signal[i][:, :(prediction_signal[x+6:x+46, y+6:y+46, 0].shape[1])]



        y_predict_hic = SA1_prediction[:,0:1,:,:]
        y_predict_signal = SA1_prediction[:,2:3,:,:]
        length = int(y_predict_hic.shape[2])
        y_predict_hic = np.reshape(y_predict_hic, (y_predict_hic.shape[0], length, length))
        for i in range(0, y_predict_hic.shape[0]):
            x = int(index_0[i][1])
            y = int(index_0[i][2])
            # prediction_hic[x:x+40, y:y+40, 1] = y_predict_hic[i]
            prediction_hic[x+6:x+46, y+6:y+46, 1] = y_predict_hic[i][:, :(prediction_hic[x+6:x+46, y+6:y+46, 1].shape[1])]

        length = int(y_predict_signal.shape[2])
        y_predict_signal = np.reshape(y_predict_signal, (y_predict_signal.shape[0], length, length))
        for i in range(0, y_predict_signal.shape[0]):
            x = int(index_0[i][1])
            y = int(index_0[i][2])
            prediction_signal[x:x+40, y:y+40, 1] = y_predict_signal[i]
            # prediction_signal[x+6:x+46, y+6:y+46, 1] = y_predict_signal[i][:, :(prediction_signal[x+6:x+46, y+6:y+46, 1].shape[1])]


        y_predict_hic = SA2_prediction[:,0:1,:,:]
        y_predict_signal = SA2_prediction[:,2:3,:,:]
        length = int(y_predict_hic.shape[2])
        y_predict_hic = np.reshape(y_predict_hic, (y_predict_hic.shape[0], length, length))
        for i in range(0, y_predict_hic.shape[0]):
            x = int(index_0[i][1])
            y = int(index_0[i][2])
            prediction_hic[x:x+40, y:y+40, 2] = y_predict_hic[i]
            # prediction_hic[x+6:x+46, y+6:y+46, 2] = y_predict_hic[i][:, :(prediction_hic[x+6:x+46, y+6:y+46, 2].shape[1])]

        length = int(y_predict_signal.shape[2])
        y_predict_signal = np.reshape(y_predict_signal, (y_predict_signal.shape[0], length, length))
        for i in range(0, y_predict_signal.shape[0]):
            x = int(index_0[i][1])
            y = int(index_0[i][2])
            prediction_signal[x:x+40, y:y+40, 2] = y_predict_signal[i]
            # prediction_signal[x+6:x+46, y+6:y+46, 2] = y_predict_signal[i][:, :(prediction_signal[x+6:x+46, y+6:y+46, 2].shape[1])]


    return(prediction_hic, prediction_signal)

# def expand_matrix(original_matrix ,i):
#     original_size = original_matrix.shape[0]
#     expanded_matrix = np.zeros((original_size + i, original_size + i))
#     expanded_matrix[:original_size, :original_size] = original_matrix
#     expanded_matrix[original_size:original_size+i, :] = 0
#     expanded_matrix[:, original_size:original_size+i] = 0
#     return expanded_matrix

def chr_pred(hicfile, chrN1, chrN2, binsize, inmodel):
    SM_M, size = utils.matrix_extract(chrN1, chrN2, binsize, f'{hicfile}/GM12878_WT_ChIAPET_SMC1A.intra_iPET_ALL.DOWNSAMPLING_43M.hic', 'SMC1A_hic', 0)
    scalerate = 16
    SM_M = (np.array(SM_M)).astype(np.float32)
    SM_N = utils.genDownsample(np.array(SM_M),1/float(scalerate))
    SM_N = (np.array(SM_N)).astype(np.float32)
    SM_O, _ = utils.matrix_extract(chrN1, chrN2, binsize, f'{hicfile}/WT_SMC1A.rmOutlier.hic', 0, size)
    SM_O = (np.array(SM_O)).astype(np.float32)


    SA1_M, _ = utils.matrix_extract(chrN1, chrN2, binsize, f'{hicfile}/GM12878_WT_ChIAPET_SA1.intra_iPET_ALL.DOWNSAMPLING_43M.hic', 0, size)
    SA1_M = (np.array(SA1_M)).astype(np.float32)
    SA1_N = utils.genDownsample(np.array(SA1_M),1/float(scalerate))
    SA1_N = (np.array(SA1_N).astype(np.float32))
    SA1_O, _ = utils.matrix_extract(chrN1, chrN2, binsize, f'{hicfile}/WT_SA1.rmOutlier.hic', 0, size)
    SA1_O = (np.array(SA1_O)).astype(np.float32)

    SA2_M, _ = utils.matrix_extract(chrN1, chrN2, binsize, f'{hicfile}/GM12878_WT_ChIAPET_SA2.intra_iPET_ALL.DOWNSAMPLING_43M.hic', 0, size)
    SA2_M = (np.array(SA2_M)).astype(np.float32)
    SA2_N = utils.genDownsample(np.array(SA2_M),1/float(scalerate))
    SA2_N = (np.array(SA2_N)).astype(np.float32)
    SA2_O, _ = utils.matrix_extract(chrN1, chrN2, binsize, f'{hicfile}/WT_SA2.rmOutlier.hic', 0, size)
    SA2_O = (np.array(SA2_O)).astype(np.float32)

    # SM = zip(utils.divide(SM_M),utils.divide(SM_N),utils.divide(SM_O))
    # SA1 = zip(utils.divide(SA1_M),utils.divide(SA1_N),utils.divide(SA1_O))
    # SA2 = zip(utils.divide(SA2_M),utils.divide(SA2_N),utils.divide(SA2_O))

    SM = np.zeros((SM_M.shape[0], SM_M.shape[0], 3))
    SM[:,:,0] = SM_M
    SM[:,:,1] = SM_N
    SM[:,:,2] = SM_O
    SA1 = np.zeros((SM_M.shape[0], SM_M.shape[0], 3))
    SA1[:,:,0] = SA1_M
    SA1[:,:,1] = SA1_N
    SA1[:,:,2] = SA1_O
    SA2 = np.zeros((SM_M.shape[0], SM_M.shape[0], 3))
    SA2[:,:,0] = SA2_M
    SA2[:,:,1] = SA2_N
    SA2[:,:,2] = SA2_O

    chr_Mat, chr_signal = predict(SM, SA1, SA2, inmodel)
    return(chr_Mat, chr_signal)

def writeBed(Mat, outname, binsize, chrN1, chrN2):
    with open(outname,'w') as chrom:
        r, c = Mat.nonzero()
        Mat = Mat * 10
        for i in range(r.size):
            contact = int(round(Mat[r[i], c[i]]))

            if contact == 0:
                continue

            line = [chrN1, r[i]*binsize, (r[i]+1)*binsize, chrN2, c[i]*binsize, (c[i]+1)*binsize, contact]
            chrom.write('chr' + str(line[0]) + ':' + str(line[1]) + '-' + str(line[2]) + '\t' + 'chr' + str(line[3]) + ':' + str(line[4]) + '-' + str(line[5]) + '\t' + str(line[6]) + '\n')

#%% 逐条染色体进行预测
### 预测 WT SMC1A
use_gpu = 1 #opt.cuda
factor = 'SMC1A'
hicfile = '/public1/chengrm/cohesin/hicplus/Input'
binsize = 5000
res = str(int(binsize/1000))
# [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 'X', 'Y']
for chr in [21]:
    print(f'============================ Predicting chr{chr} ============================')
    startTime = datetime.now()
    chrN1, chrN2 = chr, chr
    V = 'v7'
    vv = 'v13'
    inmodel = f'/public1/chengrm/cohesin/hicplus/{V}.Output/{factor}/{res}K/chr{chr}/chr{chr}_{vv}_500.model'
    outname_chr = os.path.join(f'/public1/chengrm/cohesin/hicplus/{V}.predict/{factor}/{res}K_model/')
    outname_signal = os.path.join(f'/public1/chengrm/cohesin/hicplus/{V}.predict/{factor}/{res}K_model/')

    chr_hic, chr_signal = chr_pred(hicfile, chrN1, chrN2, binsize, inmodel)
    print(chr_hic.shape, chr_signal.shape)
    writeBed(chr_hic[:,:,0], outname_chr + 'chr' + str(chrN1) + f'_SMC1A_predict_{vv}_hic_{res}kb.txt', binsize, chrN1, chrN2)
    writeBed(chr_hic[:,:,1], outname_chr + 'chr' + str(chrN1) + f'_SA1_predict_{vv}_hic_{res}kb.txt', binsize, chrN1, chrN2)
    writeBed(chr_hic[:,:,2], outname_chr + 'chr' + str(chrN1) + f'_SA2_predict_{vv}_hic_{res}kb.txt', binsize, chrN1, chrN2)

    writeBed(chr_signal[:,:,0], outname_signal + 'chr' + str(chrN1) + f'_SMC1A_predict_{vv}_signal_{res}kb.txt', binsize, chrN1, chrN2)
    writeBed(chr_signal[:,:,1], outname_signal + 'chr' + str(chrN1) + f'_SA1_predict_{vv}_signal_{res}kb.txt', binsize, chrN1, chrN2)
    writeBed(chr_signal[:,:,2], outname_signal + 'chr' + str(chrN1) + f'_SA2_predict_{vv}_signal_{res}kb.txt', binsize, chrN1, chrN2)

    time_used = datetime.now() - startTime
    print(f'chr{chr} finished, time used: {time_used}')
