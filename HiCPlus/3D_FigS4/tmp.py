from __future__ import print_function
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import utils
# import model
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from math import log10
from torch.utils import data
from time import gmtime, strftime
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed

#%% 原始数据预处理
chr_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 'X', 'Y']

human_df = utils.Hic_reader("/public1/xinyu/CohesinProject/DeepLearning_Cohesin/0.HiCPlus/3D/9_GW11B3_d20_ChIAPET_CTCF_merged_B2_3.remove_diagonal.hic", chr_list = [5])
macaque_df = utils.Hic_reader("/public1/xinyu/CohesinProject/DeepLearning_Cohesin/0.HiCPlus/3D/RheNeu_d12_ChIAPET_CTCF_merged_B1_2.remove_diagonal.hic", chr_list = [6])

## 转为2D矩阵
human_2D_matrix = utils.HiCMatrix_2D(hic_data = human_df, chr = 'chr5')
macaque_2D_matrix = utils.HiCMatrix_2D(hic_data = macaque_df, chr = 'chr6')

## 高斯平滑
human_2D_gaussian = utils.smooth_gaussian(hic_2DMat = human_2D_matrix)
macaque_2D_gaussian = utils.smooth_gaussian(hic_2DMat = macaque_2D_matrix)

## 将高斯平滑之后的矩阵转为interaction格式
bin1_idx_human, bin2_idx_human = utils.get_signal_indice(human_df, human_2D_matrix)
bin1_idx_macaque, bin2_idx_macaque = utils.get_signal_indice(macaque_df, macaque_2D_matrix)

human_gaussian_interaction = [human_2D_gaussian[bin1_idx_human[i], bin2_idx_human[i]] for i in range(len(human_df))]
macaque_gaussian_interaction = [macaque_2D_gaussian[bin1_idx_macaque[i], bin2_idx_macaque[i]] for i in range(len(macaque_df))]


human_df['gaussian'] = human_gaussian_interaction
human_df['bin1_e'] = human_df['bin1'] + 5000
human_df['bin2_e'] = human_df['bin2'] + 5000
human_df['chromosome2'] = human_df['chromosome']
human_df[['chromosome', 'bin1', 'bin1_e', 'chromosome2', 'bin2', 'bin2_e', 'gaussian']].to_csv("/public1/xinyu/CohesinProject/DeepLearning_Cohesin/0.HiCPlus/3D/Human/chr5_gasussian.bedpe", sep = '\t', index = False, header = False)


macaque_df['gaussian'] = macaque_gaussian_interaction
macaque_df['bin1_e'] = macaque_df['bin1'] + 5000
macaque_df['bin2_e'] = macaque_df['bin2'] + 5000
macaque_df['chromosome2'] = macaque_df['chromosome']
macaque_df[['chromosome', 'bin1', 'bin1_e', 'chromosome2', 'bin2', 'bin2_e', 'gaussian']].to_csv("/public1/xinyu/CohesinProject/DeepLearning_Cohesin/0.HiCPlus/3D/Macaque/chr6_gasussian.bedpe", sep = '\t', index = False, header = False)

