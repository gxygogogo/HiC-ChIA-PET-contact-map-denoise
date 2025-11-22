from __future__ import print_function
import os
import straw
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

#%% 定义hic读取函数和绘制contact map函数
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


# 绘制 Hi-C 热图
def plot_HicMatrix(chromosome, start, end, hic_data, output, vmin = 0, vmax = 10, cmap = 'viridis', sns_if = True):
    ## 读入 Hic 矩阵
    hic_count = hic_data

    ## 提取相应区域的矩阵
    hic_count_df_1 = hic_count[(hic_count['chromosome']==chromosome) & (hic_count['bin1']>=start) & (hic_count['bin2']<=end)]
    hic_count_df_1 = hic_count_df_1[['bin1', 'bin2', 'count']]
    hic_count_df_1[['count']] = hic_count_df_1[['count']].astype(int)
    hic_count_df_1.columns = [0, 1, 2]

    ## 构建对称宽数据框
    axis_all_bin = list(set([x for x in hic_count_df_1[0]] + [x for x in hic_count_df_1[1]]))
    axis_all_bin.sort()
    matrix = np.zeros((len(axis_all_bin), len(axis_all_bin)), dtype=np.float32)
    df = pd.DataFrame(matrix, index = axis_all_bin, columns = axis_all_bin)
    for i in range(len(hic_count_df_1)):
        df.loc[hic_count_df_1.iloc[i,0], hic_count_df_1.iloc[i,1]] = hic_count_df_1.iloc[i,2]
        df.loc[hic_count_df_1.iloc[i,1], hic_count_df_1.iloc[i,0]] = hic_count_df_1.iloc[i,2]

    if sns_if == True:
        # 使用 seaborn 生成热图
        plt.figure(figsize = (10, 8))
        heatmap = sns.heatmap(np.array(df), cmap = cmap, vmin = vmin, vmax = vmax)
        plt.xticks(ticks = [], labels = [])
        plt.yticks(ticks = [], labels = [])
        plt.xlabel(f'{chromosome}:{start}-{end}', fontsize = 12)
        plt.ylabel(f'{chromosome}:{start}-{end}', fontsize = 12)
        # 设置边框
        ax = heatmap.axes
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1)
        # 保存热图
        plt.savefig(output + '_sns.pdf', dpi = 400)
        # 关闭当前图像窗口
        plt.close()
    else:
        # 使用 imshow 生成热图
        plt.figure(figsize = (10, 8))
        plt.imshow(np.array(df), cmap = cmap, vmin = vmin, vmax = vmax, aspect = 'equal')
        plt.colorbar()
        plt.xticks(ticks = [], labels = [])
        plt.yticks(ticks = [], labels = [])
        plt.xlabel(f'{chromosome}:{start}-{end}', fontsize = 12)
        plt.ylabel(f'{chromosome}:{start}-{end}', fontsize = 12)
        plt.savefig(output + '_imshow.pdf', dpi = 400)
        # 关闭当前图像窗口
        plt.close()

#%% 绘图
# 参数设置
chr = 19
chromosome = 'chr19'
start = 52220001
end = 55140000

WT_SMC1A = '/data3/xiongdan/cohesin_mm10_CD8/1.downsampling/3way.NaiveCD8_WT_SMC1A_vs_SA2KO_SMC1A_vs_BRD4SA2DKO_SMC1A/NaiveCD8_WT_ChIAPET_SMC1A.intra_iPET_ALL.DOWNSAMPLING_24M.hic'

# 读取 Hic 文件
raw_count_SMC1A_df = Hic_reader(WT_SMC1A, chr_list = [19])

raw_count_SMC1A_df.columns = ['bin1', 'bin2', 'count']
raw_count_SMC1A_df['bin1'] = raw_count_SMC1A_df['bin1'].astype(int)
raw_count_SMC1A_df['bin2'] = raw_count_SMC1A_df['bin2'].astype(int)
raw_count_SMC1A_df['count'] = raw_count_SMC1A_df['count'].astype(int)


# 绘制并输出 Hic 热图
## 原始

factor = "NaiveCD8"
output = f'/public1/xinyu/CohesinProject/DeepLearning_Cohesin/0.HiCPlus/{factor}/{chromosome}'
plot_HicMatrix(chromosome = chromosome, 
               start = start, 
               end = end, 
               hic_data = raw_count_SMC1A_df, 
               vmax = 10,
               output = os.path.join(output, f'RawCount_{chromosome}_{start}_{end}_WT_{factor}_5Kb_heatmap'),
               cmap = 'YlGnBu',
               sns_if = True)


## 预测
factor = 'NaiveCD8'
output = f'/public1/xinyu/CohesinProject/DeepLearning_Cohesin/0.HiCPlus/{factor}/{chromosome}'
predict_SMC1A = pd.read_table(f'/public1/xinyu/CohesinProject/DeepLearning_Cohesin/0.HiCPlus/{factor}/{chromosome}/{chromosome}_predict_5kb.txt', sep = '\t', header = None)
predict_SMC1A[['chromosome1', 'bin1_s', 'bin1_e']] = predict_SMC1A[0].str.extract(r'(chr\d+):(\d+)-(\d+)')
predict_SMC1A[['chromosome2', 'bin2_s', 'bin2_e']] = predict_SMC1A[1].str.extract(r'(chr\d+):(\d+)-(\d+)')
predict_SMC1A_count_df = predict_SMC1A[['chromosome1', 'bin1_s', 'bin2_s', 2]]
predict_SMC1A_count_df.columns = ['chromosome', 'bin1', 'bin2', 'count']
predict_SMC1A_count_df['bin1'] = predict_SMC1A_count_df['bin1'].astype(int)
predict_SMC1A_count_df['bin2'] = predict_SMC1A_count_df['bin2'].astype(int)
predict_SMC1A_count_df['count'] = predict_SMC1A_count_df['count'].astype(float)

predict_2 = pd.merge(raw_count_SMC1A_df, predict_SMC1A_count_df, on = ['chromosome', 'bin1', 'bin2'], how = 'left')
predict_2['count_y'] = predict_2['count_y'].fillna(0)
predict_2 = predict_2[['chromosome', 'bin1', 'bin2', 'count_y']]
predict_2.columns = ['chromosome', 'bin1', 'bin2', 'count']

plot_HicMatrix(chromosome = chromosome, 
               start = start, 
               end = end, 
               hic_data = predict_2, 
               vmax = 10,
               output = os.path.join(output, f'Predict_{chromosome}_{start}_{end}_WT_{factor}_5Kb_heatmap'),
               cmap = 'YlGnBu',
               sns_if = True)

