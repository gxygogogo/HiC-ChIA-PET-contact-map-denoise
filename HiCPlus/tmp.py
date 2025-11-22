import utils
import pandas as pd
import numpy as np
from joblib import Parallel, delayed

# 读取单条染色体的hic数据
def read_single_chromosome(hic_path, chr):
    # 读取当前染色体的hic数据
    print('-------------------' + 'READING CHROMOSOME' + str(chr) + '--------------------')
    hic_tmp = straw.straw('NONE', hic_path, str(chr), str(chr), 'BP', 5000)
    # 将读取到的数据添加染色体的标签
    hic_tmp.append(['chr' + str(chr)] * len(hic_tmp[0]))
    # 将当前染色体的hic数据转换成DataFrame格式
    hic_tmp_df = pd.DataFrame(hic_tmp).T
    # 设置列名
    hic_tmp_df.columns = ['bin1', 'bin2', 'count', 'chromosome']
    return hic_tmp_df

# 读取所有染色体的hic数据
def Hic_reader(hic_path, chr_list):
    # 多核心并行读取所有染色体的数据
    hic_dfs = Parallel(n_jobs=-1)(delayed(read_single_chromosome)(hic_path, chr) for chr in chr_list)
    # 合并所有读取的数据
    hic_df = pd.concat(hic_dfs, ignore_index=True)
    return hic_df


chr_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 'X', 'Y']
raw_SMC1A = Hic_reader('/data/tang/cohesin_project/ChIA_PET_FINAL/Final.for.Downstream.Analysis/raw.data/GM12878_SA2KOA13_ChIAPET_SMC1A.intra_iPET_ALL.hic', chr_list = chr_list)
hicplus_SMC1A = Hic_reader('/public1/xinyu/CohesinProject/DeepLearning_Cohesin/0.HiCPlus/predict/SA2KO/5K/chr1_predict_5kb.hic', chr_list = [1])

tmp_df = pd.merge(raw_SMC1A, hicplus_SMC1A, on = ['bin1', 'bin2', 'chromosome'], suffixes = ('_raw', '_hicplus'))
tmp_df['bin1_e'] = tmp_df['bin1'] + 5000
tmp_df['bin2_e'] = tmp_df['bin2'] + 5000
tmp_df[['chromosome', 'bin1', 'bin1_e', 'chromosome', 'bin2', 'bin2_e', 'count_hicplus']].to_csv('/public1/xinyu/CohesinProject/DeepLearning_Cohesin/0.HiCPlus/predict/SA2KO/5K/chr1_modify_hicplus_5kb.bedpe', sep = '\t', index = False, header = False)

