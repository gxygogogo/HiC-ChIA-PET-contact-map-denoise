from __future__ import print_function
import argparse as ap
from math import log10

#import torch
#import torch.nn as nn
#import torch.optim as optim
#from torch.autograd import Variable
#from torch.utils.data import DataLoader
import utils
#import model
import argparse
import trainConvNet
import numpy as np

chrs_length = [249250621,243199373,198022430,191154276,180915260,171115067,159138663,146364022,141213431,135534747,135006516,133851895,115169878,107349540,102531392,90354753,81195210,78077248,59128983,63025520,48129895,51304566]

#chrN = 21
#scale = 16

chromosome = 1
inputfile = "/data/tang/cohesin_project/ChIA_PET_FINAL/Final.for.Downstream.Analysis/raw.data/GM12878_WT_ChIAPET_SMC1A.intra_iPET_ALL.hic"
highres = utils.train_matrix_extract(chromosome, 5000, inputfile)

print('dividing, filtering and downsampling files...')
highres_sub, index = utils.train_divide(highres)
print(highres_sub.shape)

#np.save(infile+"highres",highres_sub)
lowres = utils.genDownsample(highres,1/float(args.scalerate))
lowres_sub,index = utils.train_divide(lowres)
print(lowres_sub.shape)

#np.save(infile+"lowres",lowres_sub)
print('start training...')
trainConvNet.train(lowres_sub,highres_sub,args.outmodel)
print('finished...')