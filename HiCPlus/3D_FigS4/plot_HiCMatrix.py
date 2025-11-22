from __future__ import print_function
import os
import straw
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#%% 定义hic读取函数和绘制contact map函数
# 定义一个函数，用于读取hic文件
def Hic_reader(hic_path):
    # 定义一个列表，包含所有的染色体编号
    chr_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 'X', 'Y']
    # 创建一个空的DataFrame，用于存储读取到的hic数据
    hic_df = pd.DataFrame()

    # 遍历所有的染色体
    for chr in chr_list:
        # 读取当前染色体的hic数据
        print('-------------------' + 'READING CHROMOSOME' + str(chr) + '--------------------')
        hic_tmp = straw.straw('NONE', hic_path, str(chr), str(chr), 'BP', 5000)
        # 将读取到的数据添加到染色体的标签
        hic_tmp.append(['chr' + str(chr)]*len(hic_tmp[0]))
        # 将当前染色体的hic数据转换成DataFrame格式
        hic_tmp_df = pd.DataFrame(hic_tmp).T
        # 设置列名
        hic_tmp_df.columns = ['bin1', 'bin2', 'count', 'chromosome']
        # 将当前染色体的hic数据合并到总的hic数据框中
        hic_df = pd.concat([hic_df, hic_tmp_df], ignore_index = True)
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
        plt.xlabel(f'chr{chromosome}:{start}-{end}', fontsize = 12)
        plt.ylabel(f'chr{chromosome}:{start}-{end}', fontsize = 12)
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
        plt.xlabel(f'chr{chromosome}:{start}-{end}', fontsize = 12)
        plt.ylabel(f'chr{chromosome}:{start}-{end}', fontsize = 12)
        plt.savefig(output + '_imshow.pdf', dpi = 400)
        # 关闭当前图像窗口
        plt.close()

# 参数设置
#%% 猴子绘制
chromosome = 18
start = 5500000
end = 5700000

hicfile = "/public1/xinyu/CohesinProject/DeepLearning_Cohesin/0.HiCPlus/3D/RheNeu_d12_ChIAPET_CTCF_merged_B1_2.remove_diagonal.hic"
output = '/public1/xinyu/CohesinProject/DeepLearning_Cohesin/0.HiCPlus/3D/Macaque/remove_diagonal_5K'

# 读取 Hic 文件
hic_tmp = straw.straw('NONE', hicfile, str(chromosome), str(chromosome), 'BP', 10000)
hic_df = pd.DataFrame(hic_tmp).T
hic_df.columns = ['bin1', 'bin2', 'count']
hic_df[['chromosome']] = 'chr18'

# 绘制并输出 Hic 热图
## 原始
plot_HicMatrix(chromosome = 'chr18', 
               start = start, 
               end = end, 
               hic_data = hic_df, 
               vmax = 20,
               output = os.path.join(output, f'RawCount_{chromosome}_{start}_{end}_macaque_5Kb_heatmap'),
               cmap = 'YlGnBu',
               sns_if = True)

## 预测
predict = pd.read_table('/public1/xinyu/CohesinProject/DeepLearning_Cohesin/0.HiCPlus/3D/Macaque/remove_diagonal_5K/chr18_predict_5kb.txt', sep = '\t', header = None)
predict[['chromosome1', 'bin1_s', 'bin1_e']] = predict[0].str.extract(r'(chr\d+):(\d+)-(\d+)')
predict[['chromosome2', 'bin2_s', 'bin2_e']] = predict[1].str.extract(r'(chr\d+):(\d+)-(\d+)')
predict_count_df = predict[['chromosome1', 'bin1_s', 'bin2_s', 2]]
predict_count_df.columns = ['chromosome', 'bin1', 'bin2', 'count']
predict_count_df['bin1'] = predict_count_df['bin1'].astype(int)
predict_count_df['bin2'] = predict_count_df['bin2'].astype(int)
predict_count_df['count'] = predict_count_df['count'].astype(float)

predict = straw.straw('NONE', '/public1/xinyu/CohesinProject/DeepLearning_Cohesin/0.HiCPlus/3D/Macaque/remove_diagonal_5K/chr18_predict_10kb.hic', str(18), str(18), 'BP', 10000)
predict_df = pd.DataFrame(predict).T
predict_df.columns = ['bin1', 'bin2', 'count']
predict_df[['chromosome']] = 'chr18'


plot_HicMatrix(chromosome = 'chr18', 
               start = start, 
               end = end, 
               hic_data = predict_df, 
               vmax = 15,
               output = os.path.join(output, f'Predict_{chromosome}_{start}_{end}_macaque_5Kb_heatmap'),
               cmap = 'YlGnBu',
               sns_if = True)

#%% 人类绘制
chromosome = 18
start = 5500000
end = 5700000
hicfile = "/public1/xinyu/CohesinProject/DeepLearning_Cohesin/0.HiCPlus/3D/9_GW11B3_d20_ChIAPET_CTCF_merged_B2_3.remove_diagonal.hic"
# hicfile = "/data/yuhan/chiapet/hg38/9_GW11B3_d20_ChIAPET_CTCF_merged_B2_3/9_GW11B3_d20_ChIAPET_CTCF_merged_B2_3.hic"
output = '/public1/xinyu/CohesinProject/DeepLearning_Cohesin/0.HiCPlus/3D/Human/remove_diagonal_5K'

# 读取 Hic 文件
hic_tmp = straw.straw('NONE', hicfile, str(chromosome), str(chromosome), 'BP', 10000)
hic_df = pd.DataFrame(hic_tmp).T
hic_df.columns = ['bin1', 'bin2', 'count']
hic_df[['chromosome']] = 'chr5'

# 绘制并输出 Hic 热图
## 原始
plot_HicMatrix(chromosome = chromosome, 
               start = start, 
               end = end, 
               hic_data = hic_df, 
               vmax = 20,
               output = os.path.join(output, f'RawCount_{chromosome}_{start}_{end}_human_5Kb_heatmap'),
               cmap = 'YlGnBu',
               sns_if = True)

## 预测
predict = pd.read_table('/public1/xinyu/CohesinProject/DeepLearning_Cohesin/0.HiCPlus/3D/Human/remove_diagonal_5K/chr5_predict_5.0kb.txt', sep = '\t', header = None)
predict[['chromosome1', 'bin1_s', 'bin1_e']] = predict[0].str.extract(r'(chr\d+):(\d+)-(\d+)')
predict[['chromosome2', 'bin2_s', 'bin2_e']] = predict[1].str.extract(r'(chr\d+):(\d+)-(\d+)')
predict_count_df = predict[['chromosome1', 'bin1_s', 'bin2_s', 2]]
predict_count_df.columns = ['chromosome', 'bin1', 'bin2', 'count']
predict_count_df['bin1'] = predict_count_df['bin1'].astype(int)
predict_count_df['bin2'] = predict_count_df['bin2'].astype(int)
predict_count_df['count'] = predict_count_df['count'].astype(float)

predict = straw.straw('NONE', '/public1/xinyu/CohesinProject/DeepLearning_Cohesin/0.HiCPlus/3D/Macaque/remove_diagonal_5K/chr18_predict_10kb.hic', str(18), str(18), 'BP', 10000)
predict_df = pd.DataFrame(predict).T
predict_df.columns = ['bin1', 'bin2', 'count']
predict_df[['chromosome']] = 'chr18'

plot_HicMatrix(chromosome = 'chr5', 
               start = start, 
               end = end, 
               hic_data = predict_df, 
               vmax = 20,
               output = os.path.join(output, f'Predict_{chromosome}_{start}_{end}_human_5Kb_heatmap'),
               cmap = 'YlGnBu',
               sns_if = True)
