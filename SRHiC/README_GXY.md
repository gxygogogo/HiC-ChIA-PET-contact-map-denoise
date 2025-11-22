# 直接使用文章的训练模型
## step1: 获取Valid pair格式文件
使用run.get_Valid_pair.sh脚本获得Valid pair格式文件，用于生成SRHiC预测所需的数据格式

## step2：生成reads文件
使用step1生成的Valid pair格式文件，利用run.chr_reads.py生成reads文件，用于生成SRHiC预测所需的数据格式

## step3：生成SRHiC预测所需的数据格式
使用step2生成的reads文件，利用run.get_input_Format.py生成预测所需的数据格式

## step4：利用训练好的模型进行预测
使用SRHiC_main.py进行预测，得到subMats的结果

## step5：将subMats整合到一起
使用run.combine_subMat.py将预测得到的subMats整合到一起

## step6：预测结果可视化
使用run.heatmap_plot.R将merged Matrix进行可视化

