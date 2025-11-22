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
chr = 1
chromosome = 'chr1'
start = 224610000
end = 224860000

hicfile = '/public1/xinyu/CohesinProject/DeepLearning_Cohesin/0.HiCPlus/Input/GM12878_WT_ChIAPET_SA2CST.intra_iPET_ALL.hic'
SA1_hicfile = '/public1/xinyu/CohesinProject/DeepLearning_Cohesin/0.HiCPlus/Input/GM12878_WT_ChIAPET_SA1.intra_iPET_ALL.hic'
SMC1A_hicfile = '/public1/xinyu/CohesinProject/DeepLearning_Cohesin/0.HiCPlus/Input/GM12878_WT_ChIAPET_SMC1A.intra_iPET_ALL.hic' 
SA1KO_hicfile = '/public1/xinyu/CohesinProject/DeepLearning_Cohesin/0.HiCPlus/Input/GM12878_SA1KO_ChIAPET_SMC1A.intra_iPET_ALL.hic'
SA2KO_hicfile = '/public1/xinyu/CohesinProject/DeepLearning_Cohesin/0.HiCPlus/Input/GM12878_SA2KOA13_ChIAPET_SMC1A.intra_iPET_ALL.hic'


# 读取 Hic 文件
raw_count_df = Hic_reader(hicfile)

raw_count_SA1_df = pd.DataFrame(straw.straw('NONE', SA1_hicfile, str(chr), str(chr), 'BP', 5000)).T
raw_count_SA1_df.columns = ['bin1', 'bin2', 'count']
raw_count_SA1_df['bin1'] = raw_count_SA1_df['bin1'].astype(int)
raw_count_SA1_df['bin2'] = raw_count_SA1_df['bin2'].astype(int)
raw_count_SA1_df['count'] = raw_count_SA1_df['count'].astype(int)
raw_count_SA1_df[['chromosome']] = chromosome

raw_count_SMC1A_df = pd.DataFrame(straw.straw('NONE', SMC1A_hicfile, str(chr), str(chr), 'BP', 5000)).T
raw_count_SMC1A_df.columns = ['bin1', 'bin2', 'count']
raw_count_SMC1A_df['bin1'] = raw_count_SMC1A_df['bin1'].astype(int)
raw_count_SMC1A_df['bin2'] = raw_count_SMC1A_df['bin2'].astype(int)
raw_count_SMC1A_df['count'] = raw_count_SMC1A_df['count'].astype(int)
raw_count_SMC1A_df[['chromosome']] = chromosome

raw_count_SA1KO_df = pd.DataFrame(straw.straw('NONE', SA1KO_hicfile, str(chr), str(chr), 'BP', 5000)).T
raw_count_SA1KO_df.columns = ['bin1', 'bin2', 'count']
raw_count_SA1KO_df['bin1'] = raw_count_SA1KO_df['bin1'].astype(int)
raw_count_SA1KO_df['bin2'] = raw_count_SA1KO_df['bin2'].astype(int)
raw_count_SA1KO_df['count'] = raw_count_SA1KO_df['count'].astype(int)
raw_count_SA1KO_df[['chromosome']] = chromosome

raw_count_SA2KO_df = pd.DataFrame(straw.straw('NONE', SA2KO_hicfile, str(chr), str(chr), 'BP', 5000)).T
raw_count_SA2KO_df.columns = ['bin1', 'bin2', 'count']
raw_count_SA2KO_df['bin1'] = raw_count_SA2KO_df['bin1'].astype(int)
raw_count_SA2KO_df['bin2'] = raw_count_SA2KO_df['bin2'].astype(int)
raw_count_SA2KO_df['count'] = raw_count_SA2KO_df['count'].astype(int)
raw_count_SA2KO_df[['chromosome']] = chromosome


# 绘制并输出 Hic 热图
## 原始
factor = "SA2"
output = f'/public1/xinyu/CohesinProject/DeepLearning_Cohesin/0.HiCPlus/predict/{factor}/5K'
plot_HicMatrix(chromosome = chromosome, 
               start = start, 
               end = end, 
               hic_data = raw_count_df, 
               vmax = 10,
               output = os.path.join(output, f'RawCount_{chromosome}_{start}_{end}_WT_{factor}_5Kb_heatmap'),
               cmap = 'YlGnBu',
               sns_if = True)

factor = "SA1"
output = f'/public1/xinyu/CohesinProject/DeepLearning_Cohesin/0.HiCPlus/predict/{factor}/5K'
plot_HicMatrix(chromosome = chromosome, 
               start = start, 
               end = end, 
               hic_data = raw_count_SA1_df, 
               vmax = 10,
               output = os.path.join(output, f'RawCount_{chromosome}_{start}_{end}_WT_{factor}_5Kb_heatmap'),
               cmap = 'YlGnBu',
               sns_if = True)

factor = "SMC1A"
output = f'/public1/xinyu/CohesinProject/DeepLearning_Cohesin/0.HiCPlus/predict/{factor}/5K_model'
plot_HicMatrix(chromosome = chromosome, 
               start = start, 
               end = end, 
               hic_data = raw_count_SMC1A_df, 
               vmax = 10,
               output = os.path.join(output, f'RawCount_{chromosome}_{start}_{end}_WT_{factor}_5Kb_heatmap'),
               cmap = 'YlGnBu',
               sns_if = True)

factor = "SA1KO"
output = f'/public1/xinyu/CohesinProject/DeepLearning_Cohesin/0.HiCPlus/predict/{factor}/5K'
plot_HicMatrix(chromosome = chromosome, 
               start = start, 
               end = end, 
               hic_data = raw_count_SA1KO_df, 
               vmax = 10,
               output = os.path.join(output, f'RawCount_{chromosome}_{start}_{end}_WT_{factor}_5Kb_heatmap'),
               cmap = 'YlGnBu',
               sns_if = True)

factor = "SA2KO"
output = f'/public1/xinyu/CohesinProject/DeepLearning_Cohesin/0.HiCPlus/predict/{factor}/5K'
plot_HicMatrix(chromosome = chromosome, 
               start = start, 
               end = end, 
               hic_data = raw_count_SA2KO_df, 
               vmax = 10,
               output = os.path.join(output, f'RawCount_{chromosome}_{start}_{end}_WT_{factor}_5Kb_heatmap'),
               cmap = 'YlGnBu',
               sns_if = True)


## 预测
# predict_file = f'/public1/xinyu/CohesinProject/DeepLearning_Cohesin/0.HiCPlus/predict/{factor}/5K/chr21_predict_5kb.hic'
factor = 'SA2'
output = f'/public1/xinyu/CohesinProject/DeepLearning_Cohesin/0.HiCPlus/predict/{factor}/5K'
predict_SA2 = pd.read_table(f'/public1/xinyu/CohesinProject/DeepLearning_Cohesin/0.HiCPlus/predict/{factor}/5K/{chromosome}_predict_5kb.txt', sep = '\t', header = None)
predict_SA2[['chromosome1', 'bin1_s', 'bin1_e']] = predict_SA2[0].str.extract(r'(chr\d+):(\d+)-(\d+)')
predict_SA2[['chromosome2', 'bin2_s', 'bin2_e']] = predict_SA2[1].str.extract(r'(chr\d+):(\d+)-(\d+)')
predict_SA2_count_df = predict_SA2[['chromosome1', 'bin1_s', 'bin2_s', 2]]
predict_SA2_count_df.columns = ['chromosome', 'bin1', 'bin2', 'count']
predict_SA2_count_df['bin1'] = predict_SA2_count_df['bin1'].astype(int)
predict_SA2_count_df['bin2'] = predict_SA2_count_df['bin2'].astype(int)
predict_SA2_count_df['count'] = predict_SA2_count_df['count'].astype(float)
plot_HicMatrix(chromosome = chromosome, 
               start = start, 
               end = end, 
               hic_data = predict_SA2_count_df, 
               vmax = 10,
               output = os.path.join(output, f'Predict_{chromosome}_{start}_{end}_WT_{factor}_5Kb_heatmap'),
               cmap = 'YlGnBu',
               sns_if = True)


factor = 'SA1'
output = f'/public1/xinyu/CohesinProject/DeepLearning_Cohesin/0.HiCPlus/predict/{factor}/5K'
predict_SA1 = pd.read_table(f'/public1/xinyu/CohesinProject/DeepLearning_Cohesin/0.HiCPlus/predict/{factor}/5K/{chromosome}_predict_5kb.txt', sep = '\t', header = None)
predict_SA1[['chromosome1', 'bin1_s', 'bin1_e']] = predict_SA1[0].str.extract(r'(chr\d+):(\d+)-(\d+)')
predict_SA1[['chromosome2', 'bin2_s', 'bin2_e']] = predict_SA1[1].str.extract(r'(chr\d+):(\d+)-(\d+)')
predict_SA1_count_df = predict_SA1[['chromosome1', 'bin1_s', 'bin2_s', 2]]
predict_SA1_count_df.columns = ['chromosome', 'bin1', 'bin2', 'count']
predict_SA1_count_df['bin1'] = predict_SA1_count_df['bin1'].astype(int)
predict_SA1_count_df['bin2'] = predict_SA1_count_df['bin2'].astype(int)
predict_SA1_count_df['count'] = predict_SA1_count_df['count'].astype(float)
plot_HicMatrix(chromosome = chromosome, 
               start = start, 
               end = end, 
               hic_data = predict_SA1_count_df, 
               vmax = 10,
               output = os.path.join(output, f'Predict_{chromosome}_{start}_{end}_WT_{factor}_5Kb_heatmap'),
               cmap = 'YlGnBu',
               sns_if = True)


factor = 'SMC1A'
output = f'/public1/xinyu/CohesinProject/DeepLearning_Cohesin/0.HiCPlus/predict/{factor}/5K_model'
predict_SMC1A = pd.read_table(f'/public1/xinyu/CohesinProject/DeepLearning_Cohesin/0.HiCPlus/predict/{factor}/5K_model/{chromosome}_predict_5kb.txt', sep = '\t', header = None)
predict_SMC1A[['chromosome1', 'bin1_s', 'bin1_e']] = predict_SMC1A[0].str.extract(r'(chr\d+):(\d+)-(\d+)')
predict_SMC1A[['chromosome2', 'bin2_s', 'bin2_e']] = predict_SMC1A[1].str.extract(r'(chr\d+):(\d+)-(\d+)')
predict_SMC1A_count_df = predict_SMC1A[['chromosome1', 'bin1_s', 'bin2_s', 2]]
predict_SMC1A_count_df.columns = ['chromosome', 'bin1', 'bin2', 'count']
predict_SMC1A_count_df['bin1'] = predict_SMC1A_count_df['bin1'].astype(int)
predict_SMC1A_count_df['bin2'] = predict_SMC1A_count_df['bin2'].astype(int)
predict_SMC1A_count_df['count'] = predict_SMC1A_count_df['count'].astype(float)
plot_HicMatrix(chromosome = chromosome, 
               start = start, 
               end = end, 
               hic_data = predict_SMC1A_count_df, 
               vmax = 10,
               output = os.path.join(output, f'Predict_{chromosome}_{start}_{end}_WT_{factor}_5Kb_heatmap'),
               cmap = 'YlGnBu',
               sns_if = True)


factor = 'SA1KO'
output = f'/public1/xinyu/CohesinProject/DeepLearning_Cohesin/0.HiCPlus/predict/{factor}/5K'
predict_SA1KO = pd.read_table(f'/public1/xinyu/CohesinProject/DeepLearning_Cohesin/0.HiCPlus/predict/{factor}/5K/{chromosome}_predict_5kb.txt', sep = '\t', header = None)
predict_SA1KO[['chromosome1', 'bin1_s', 'bin1_e']] = predict_SA1KO[0].str.extract(r'(chr\d+):(\d+)-(\d+)')
predict_SA1KO[['chromosome2', 'bin2_s', 'bin2_e']] = predict_SA1KO[1].str.extract(r'(chr\d+):(\d+)-(\d+)')
predict_SA1KO_count_df = predict_SA1KO[['chromosome1', 'bin1_s', 'bin2_s', 2]]
predict_SA1KO_count_df.columns = ['chromosome', 'bin1', 'bin2', 'count']
predict_SA1KO_count_df['bin1'] = predict_SA1KO_count_df['bin1'].astype(int)
predict_SA1KO_count_df['bin2'] = predict_SA1KO_count_df['bin2'].astype(int)
predict_SA1KO_count_df['count'] = predict_SA1KO_count_df['count'].astype(float)
plot_HicMatrix(chromosome = chromosome, 
               start = start, 
               end = end, 
               hic_data = predict_SA1KO_count_df, 
               vmax = 10,
               output = os.path.join(output, f'Predict_{chromosome}_{start}_{end}_WT_{factor}_5Kb_heatmap'),
               cmap = 'YlGnBu',
               sns_if = True)


factor = 'SA2KO'
output = f'/public1/xinyu/CohesinProject/DeepLearning_Cohesin/0.HiCPlus/predict/{factor}/5K'
predict_SA2KO = pd.read_table(f'/public1/xinyu/CohesinProject/DeepLearning_Cohesin/0.HiCPlus/predict/{factor}/5K/{chromosome}_predict_5kb.txt', sep = '\t', header = None)
predict_SA2KO[['chromosome1', 'bin1_s', 'bin1_e']] = predict_SA2KO[0].str.extract(r'(chr\d+):(\d+)-(\d+)')
predict_SA2KO[['chromosome2', 'bin2_s', 'bin2_e']] = predict_SA2KO[1].str.extract(r'(chr\d+):(\d+)-(\d+)')
predict_SA2KO_count_df = predict_SA2KO[['chromosome1', 'bin1_s', 'bin2_s', 2]]
predict_SA2KO_count_df.columns = ['chromosome', 'bin1', 'bin2', 'count']
predict_SA2KO_count_df['bin1'] = predict_SA2KO_count_df['bin1'].astype(int)
predict_SA2KO_count_df['bin2'] = predict_SA2KO_count_df['bin2'].astype(int)
predict_SA2KO_count_df['count'] = predict_SA2KO_count_df['count'].astype(float)
plot_HicMatrix(chromosome = chromosome, 
               start = start, 
               end = end, 
               hic_data = predict_SA2KO_count_df, 
               vmax = 10,
               output = os.path.join(output, f'Predict_{chromosome}_{start}_{end}_WT_{factor}_5Kb_heatmap'),
               cmap = 'YlGnBu',
               sns_if = True)

# predict_count_df = Hic_reader(predict)






# %%
