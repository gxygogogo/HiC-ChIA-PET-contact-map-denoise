#%% 加载库
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
def predict(M, N, inmodel):
    prediction_1 = np.zeros((N, N))
    for low_resolution_samples, index in utils.divide(M):

        batch_size = low_resolution_samples.shape[0] #256
        lowres_set = data.TensorDataset(torch.from_numpy(low_resolution_samples), torch.from_numpy(np.zeros(low_resolution_samples.shape[0])))
        try:
            lowres_loader = torch.utils.data.DataLoader(lowres_set, batch_size = batch_size, shuffle = False)
        except:
            continue

        hires_loader = lowres_loader
        m = model.Net(40, 28)
        m.load_state_dict(torch.load(inmodel, map_location = torch.device('cpu')))
        if torch.cuda.is_available():
            m = m.cuda()
            use_gpu = True
        else:
            use_gpu = False

        for i, v1 in enumerate(lowres_loader):
            _lowRes, _ = v1
            _lowRes = Variable(_lowRes).float()
            if use_gpu:
                _lowRes = _lowRes.cuda()
            y_prediction = m(_lowRes)

        y_predict = y_prediction.data.cpu().numpy()

        # recombine samples
        length = int(y_predict.shape[2])
        y_predict = np.reshape(y_predict, (y_predict.shape[0], length, length))

        for i in range(0, y_predict.shape[0]):
            x = int(index[i][1])
            y = int(index[i][2])

            prediction_1[x+6:x+34, y+6:y+34] = y_predict[i]
    return(prediction_1)

def chr_pred(hicfile, chrN1, chrN2, binsize, inmodel):
    M = utils.matrix_extract(chrN1, chrN2, binsize, hicfile)
    N = M.shape[0]
    chr_Mat = predict(M, N, inmodel)
    return(chr_Mat)

def writeBed(Mat, outname, binsize, chrN1, chrN2):
    with open(outname,'w') as chrom:
        r, c = Mat.nonzero()
        for i in range(r.size):
            contact = int(round(Mat[r[i], c[i]]))
            if contact == 0:
                continue

            line = [chrN1, r[i]*binsize, (r[i]+1)*binsize, chrN2, c[i]*binsize, (c[i]+1)*binsize, contact]
            chrom.write('chr' + str(line[0]) + ':' + str(line[1]) + '-' + str(line[2]) + '\t' + 'chr' + str(line[3]) + ':' + str(line[4]) + '-' + str(line[5]) + '\t' + str(line[6]) + '\n')

#%% 逐条染色体进行预测
### 增强 Human neuron CTCF
startTime = datetime.now()
use_gpu = 1 #opt.cuda
hicfile = '/public1/xinyu/CohesinProject/DeepLearning_Cohesin/0.HiCPlus/3D/9_GW11B3_d20_ChIAPET_CTCF_merged_B2_3.remove_diagonal.hic'
binsize = 10000
res = binsize/1000

chrN1, chrN2 = 18, 18
inmodel = f'/public1/xinyu/CohesinProject/DeepLearning_Cohesin/0.HiCPlus/3D/Human/remove_diagonal_5K/chr{chrN1}_3400.model'
outname = os.path.join(f'/public1/xinyu/CohesinProject/DeepLearning_Cohesin/0.HiCPlus/3D/Human/remove_diagonal_5K', 'chr' + str(chrN1) + f'_predict_{res}kb.txt')
Mat = chr_pred(hicfile, chrN1, chrN2, binsize, inmodel)
print(Mat.shape)
writeBed(Mat, outname, binsize, chrN1, chrN2)

print(datetime.now() - startTime)


### 增强 Macaque neuron CTCF
startTime = datetime.now()
use_gpu = 1 #opt.cuda
hicfile = '/public1/xinyu/CohesinProject/DeepLearning_Cohesin/0.HiCPlus/3D/RheNeu_d12_ChIAPET_CTCF_merged_B1_2.remove_diagonal.hic'
binsize = 10000
res = str(int(binsize/1000))

chrN1, chrN2 = 18, 18
inmodel = f'/public1/xinyu/CohesinProject/DeepLearning_Cohesin/0.HiCPlus/3D/Macaque/remove_diagonal_5K/chr{chrN1}_3400.model'
outname = os.path.join(f'/public1/xinyu/CohesinProject/DeepLearning_Cohesin/0.HiCPlus/3D/Macaque/remove_diagonal_5K', 'chr' + str(chrN1) + f'_predict_{res}kb.txt')
Mat = chr_pred(hicfile, chrN1, chrN2, binsize, inmodel)
print(Mat.shape)
writeBed(Mat, outname, binsize, chrN1, chrN2)

print(datetime.now() - startTime)

