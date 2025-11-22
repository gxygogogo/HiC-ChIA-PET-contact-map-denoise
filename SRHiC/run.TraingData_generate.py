import numpy as np
import pandas as pd
import sys
import math

sub_mat_n = 40
step = 28




## hg38 chromosome length
chr_len_list = pd.read_table('/data/public/refGenome/bwa_index/hg38/hg38.chrom.sizes', sep = '\t', header = None)
chr_len_list.columns = ['chr', 'chr_len']

file_Mat         = sys.argv[1]
chr_len          = int(sys.argv[2])
resolution       = int(sys.argv[3])
file_out_subMats = sys.argv[4]
file_out_index   = sys.argv[5]

num_bins = math.ceil(chr_len / resolution)
chrMat = np.zeros((num_bins, num_bins)).astype(np.int16)
# read pairs
filein = open(file_Mat).readlines()
for i in range(0, len(filein)):
    items = filein[i].split()
    i1 = math.ceil(int(items[0]) / resolution)
    i2 = math.ceil(int(items[1]) / resolution)
    chrMat[i1][i2] = float(items[2])
    chrMat[i2][i1] = float(items[2])

subMats = []
index = []
for i in range(0, num_bins, step):
	for j in range(i, num_bins, step):
	  # abs(j-i)>201 means that we only care about genomic distance <= 2Mb
		if (i + sub_mat_n >= num_bins or j + sub_mat_n >= num_bins or abs(j-i) > 201):
			continue
		subMat = chrMat[i:i+sub_mat_n, j:j+sub_mat_n]
		subMats.append([subMat,])
		index.append((i+6, j+6))

index = np.array(index)
subMats = np.array(subMats)
subMats = subMats.astype(np.double)



np.save(file_out_subMats, subMats)
np.save(file_out_index, index)


