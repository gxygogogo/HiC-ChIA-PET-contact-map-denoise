# Contact Map 热图去噪
## HiCPlus
### Step1: 训练
直接使用train_models.py脚本对hic文件进行训练
### Step2: 预测
基于上一步训练好的模型，使用pred_chromosome.py进行逐条染色体预测，最后回拼为全基因组热图
### Step3: 绘图
使用plot_HicMatrix.py进行Contact Map热图绘制

![HiCPlus](https://github.com/gxygogogo/HiC-ChIA-PET-contact-map-denoise/blob/main/result/HiCPlus.png)

## SRHiC
### step1: 获取Valid pair格式文件
直接使用文章的训练模型，使用run.get_Valid_pair.sh脚本获得Valid pair格式文件，用于生成SRHiC预测所需的数据格式

### step2：生成reads文件
使用step1生成的Valid pair格式文件，利用run.chr_reads.py生成reads文件，用于生成SRHiC预测所需的数据格式

### step3：生成SRHiC预测所需的数据格式
使用step2生成的reads文件，利用run.get_input_Format.py生成预测所需的数据格式

### step4：利用训练好的模型进行预测
使用SRHiC_main.py进行预测，得到subMats的结果

### step5：将subMats整合到一起
使用run.combine_subMat.py将预测得到的subMats整合到一起

### step6：预测结果可视化
使用run.heatmap_plot.R将merged Matrix进行可视化
![SRHiC](https://github.com/gxygogogo/HiC-ChIA-PET-contact-map-denoise/blob/main/result/SRHiC.png)
