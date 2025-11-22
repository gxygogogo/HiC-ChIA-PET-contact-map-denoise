import sys
import numpy as np
import pandas as pd


dat_predict = np.squeeze(np.load("/public1/xinyu/CohesinProject/SRHiC/predict/WT_SMC1A/5K_removeDiagonal/enhanced_chr16_remove_Diagonal.subMats.npy").astype(np.float32))
dat_index   = np.load("/public1/xinyu/CohesinProject/SRHiC/data/inHouse_test_tmp/WT_SMC1A/5K/chr16_remove_Diagonal.index.npy")
chr_len     = int(90338345)
resolution  = int(5000)
file_output = "/public1/xinyu/CohesinProject/SRHiC/predict/WT_SMC1A/5K_removeDiagonal/enhanced_chr16_remove_Diagonal.totalMats.npy"

num_bins = np.ceil(chr_len / resolution).astype('int')
mat = np.zeros((num_bins, num_bins))

for i in range(dat_predict.shape[0]):
	r1 = dat_index[i,0]
	c1 = dat_index[i,1]
	r2 = r1 + 27 + 1
	c2 = c1 + 27 + 1
	mat[r1:r2, c1:c2] = dat_predict[i,:,:]


# copy upper triangle to lower triangle
lower_index = np.tril_indices(num_bins, -1)
mat[lower_index] = mat.T[lower_index]  


np.save(file_output, mat)

mat = np.load("/public1/xinyu/CohesinProject/SRHiC/predict/WT_SMC1A/5K_removeDiagonal/enhanced_chr16_remove_Diagonal.totalMats.npy", allow_pickle=True)
mat_df = pd.DataFrame(mat)
mat_df.to_csv('/public1/xinyu/CohesinProject/SRHiC/predict/WT_SMC1A/5K_removeDiagonal/enhanced_chr16_remove_Diagonal.totalMats.txt', sep = "\t", header = False, index = False)

