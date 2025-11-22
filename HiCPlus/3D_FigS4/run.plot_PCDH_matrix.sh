#!/bin/bash
juicer=/data/public/software/juicer_tools/juicer_tools.1.6.1_jcuda.0.7.5.jar
alias hicConvertFormat=/home/xiongdan/software/HiCExplorer-3.7.2/bin/hicConvertFormat
alias hicPlotMatrix=/home/xiongdan/software/HiCExplorer-3.7.2/bin/hicPlotMatrix

#################################################################################################################################################################
## Human
## 转为bedpe
cat chr5_predict_5.0kb.txt | sed s'/[:-]/\t/'g > Human_chr5_enhanced.bedpe

## 转为hic
cat chr5_gasussian.bedpe | awk '{print 0,$1,$2,0,0,$4,$5,1,$7}' >  chr5_gasussian_tmp.txt
less chr5_gasussian_tmp.txt |awk '{OFS="\t"; for (i=0; i<$NF; i++) print }' |cut -f1-8 -d" " > chr5_gasussian_tmp2.txt
java -Xmx20g -jar ${juicer} pre -r 5000,10000,20000,25000,40000,50000,100000 chr5_gasussian_tmp2.txt chr5_gasussian.hic hg38

## 转为h5
hicConvertFormat -m chr5_gasussian.hic --inputFormat hic -o chr5_gasussian.cool --outputFormat cool --chromosome 5 --resolutions 5000
hicConvertFormat -m chr5_gasussian_5000.cool --inputFormat cool -o chr5_gasussian.h5 --outputFormat h5 --resolutions 5000


## 转为hic
cat Human_chr5_enhanced.bedpe | awk '{print 0,$1,$2,0,0,$4,$5,1,$7}' >  Human_chr5_enhanced_tmp.txt
less Human_chr5_enhanced_tmp.txt |awk '{OFS="\t"; for (i=0; i<$NF; i++) print }' |cut -f1-8 -d" " > Human_chr5_enhanced_tmp2.txt
java -Xmx20g -jar ${juicer} pre -r 5000,10000,20000,25000,40000,50000,100000 Human_chr5_enhanced_tmp2.txt Human_chr5_enhanced.hic hg38

## 转为h5
hicConvertFormat -m Human_chr5_enhanced.hic --inputFormat hic -o Human_chr5_enhanced.cool --outputFormat cool --chromosome 5 --resolutions 5000
hicConvertFormat -m Human_chr5_enhanced_5000.cool --inputFormat cool -o Human_chr5_enhanced.h5 --outputFormat h5 --resolutions 5000



## 绘图
region=chr5:141047993-141271393
region2=chr5:141271393-141593466
res=5000
chr=5
r=$((${res}/1000))
name=chr5_gasussian
hicPlotMatrix --colorMap Greens --vMin 0.5 --vMax 3 --region ${region} --region2 ${region2} -m chr5_gasussian.h5 -o ${name}.${r}kb.gaussian.${region}_${region2}.pdf
hicPlotMatrix --colorMap Greens --vMin 0.5 --vMax 6 --region ${region} --region2 ${region2} -m chr5_gasussian.h5 -o ${name}.${r}kb.gaussian.${region}_${region2}.pdf
hicPlotMatrix --colorMap Greens --vMin 0.5 --vMax 10 --region ${region} --region2 ${region2} -m chr5_gasussian.h5 -o ${name}.${r}kb.gaussian.${region}_${region2}.pdf


#################################################################################################################################################################
## Macaque
## 转为bedpe
cat chr6_predict_5kb.txt | sed s'/[:-]/\t/'g > Macaque_chr6_enhanced.bedpe

## 转为hic
chr_size=/data2/yuhan/Project/3D_evolution/HiCDCPlus/macaque/chr_size.txt
cat Macaque_chr6_enhanced.bedpe | awk '{print 0,$1,$2,0,0,$4,$5,1,$7}' >  Macaque_chr6_enhanced_tmp.txt
less Macaque_chr6_enhanced_tmp.txt |awk '{OFS="\t"; for (i=0; i<$NF; i++) print }' |cut -f1-8 -d" " > Macaque_chr6_enhanced_tmp2.txt
java -Xmx20g -jar ${juicer} pre -r 5000,10000,20000,25000,40000,50000,100000 Macaque_chr6_enhanced_tmp2.txt Macaque_chr6_enhanced.hic ${chr_size}

## 转为h5
hicConvertFormat -m Macaque_chr6_enhanced.hic --inputFormat hic -o Macaque_chr6_enhanced.cool --outputFormat cool --chromosome 6 --resolutions 5000
hicConvertFormat -m Macaque_chr6_enhanced_5000.cool --inputFormat cool -o Macaque_chr6_enhanced.h5 --outputFormat h5 --resolutions 5000


## 转为hic
chr_size=/data2/yuhan/Project/3D_evolution/HiCDCPlus/macaque/chr_size.txt
cat chr6_gasussian.bedpe | awk '{print 0,$1,$2,0,0,$4,$5,1,$7}' >  chr6_gasussian_tmp.txt
less chr6_gasussian_tmp.txt |awk '{OFS="\t"; for (i=0; i<$NF; i++) print }' |cut -f1-8 -d" " > chr6_gasussian_tmp2.txt
java -Xmx20g -jar ${juicer} pre -r 5000,10000,20000,25000,40000,50000,100000 chr6_gasussian_tmp2.txt chr6_gasussian.hic ${chr_size}

## 转为h5
hicConvertFormat -m chr6_gasussian.hic --inputFormat hic -o chr6_gasussian.cool --outputFormat cool --chromosome 6 --resolutions 5000
hicConvertFormat -m chr6_gasussian_5000.cool --inputFormat cool -o chr6_gasussian.h5 --outputFormat h5 --resolutions 5000


## 绘图
region=chr6:138814038-139029543
region2=chr6:139029543-139367102
res=5000
chr=6
r=$((${res}/1000))
name=Macaque_chr6_enhanced
hicPlotMatrix --colorMap Greens --vMin 0.5 --vMax 3 --region ${region} --region2 ${region2} -m Macaque_chr6_enhanced.h5 -o ${name}.${r}kb.enhance.${region}_${region2}.pdf
hicPlotMatrix --colorMap Greens --vMin 0.5 --vMax 6 --region ${region} --region2 ${region2} -m Macaque_chr6_enhanced.h5 -o ${name}.${r}kb.enhance.${region}_${region2}.pdf
hicPlotMatrix --colorMap Greens --vMin 0.5 --vMax 10 --region ${region} --region2 ${region2} -m Macaque_chr6_enhanced.h5 -o ${name}.${r}kb.enhance.${region}_${region2}.pdf
