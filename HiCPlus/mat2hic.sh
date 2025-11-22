#/bin/bash


# dat=$1 ##output from hicplus prediction

# chrom=hg38.chrom.sizes ##chrom size file, change to your own species.

# cat $dat | tr ':' '\t'|tr '-' '\t' | awk '{print 0,$1,$2,0,0,$4,$5,1,$7}' >  ${dat}_tmp


# less ${dat}_tmp |awk '{OFS="\t"; for (i=0; i<$NF; i++) print }' |cut -f1-8 -d" " > ${dat}_tmp2

# java -Xmx40g -jar ~/jwn2291/juicer/scripts/scripts/juicer_tools_1.13.02.jar pre -d -c 18 -r 5000,10000,20000,25000,40000,50000,100000 ${dat}_tmp2 ${dat}.hic hg38

juicer=/data/public/software/juicer_tools/juicer_tools.1.6.1_jcuda.0.7.5.jar

## 将单条染色体的预测结果整合到一起
for i in {1..22} X Y
do
    echo "chr"$i
    cat chr${i}_predict_5.0kb.txt >> WT_SMC1A_predict_5kb.txt
done

cat WT_SMC1A_predict_5kb.txt | sed s'/[:-]/\t/'g | LANG=C sort -k1,1 -k2,2d -k6,6d > WT_SMC1A_predict_5kb.bedpe

## 创建HiC文件
cat WT_SMC1A_modify_5kb.bedpe | awk '{print 0,$1,$2,0,0,$4,$5,1,$7}' >  WT_SMC1A_modify_5kb_tmp.txt
less WT_SMC1A_modify_5kb_tmp.txt |awk '{OFS="\t"; for (i=0; i<$NF; i++) print }' |cut -f1-8 -d" " > WT_SMC1A_modify_5kb_tmp2.txt

java -Xmx20g -jar ${juicer} pre -r 5000,10000,20000,25000,40000,50000,100000 WT_SMC1A_modify_5kb_tmp.txt WT_SMC1A_modify_5kb.V2.hic hg38




cat chr1_predict_5kb.txt | sed s'/[:-]/\t/'g | LANG=C sort -k1,1 -k2,2d -k6,6d > chr1_predict_5kb.bedpe
cat chr1_predict_5kb.bedpe | awk '{print 0,$1,$2,0,0,$4,$5,1,$7}' >  chr1_predict_5kb_tmp.txt
less chr1_predict_5kb_tmp.txt |awk '{OFS="\t"; for (i=0; i<$NF; i++) print }' |cut -f1-8 -d" " > chr1_predict_5kb_tmp2.txt
java -Xmx20g -jar ${juicer} pre -r 5000,10000,20000,25000,40000,50000,100000 chr1_predict_5kb_tmp.txt chr1_predict_5kb.hic hg38



## 单条染色体转为 HiC 文件
cat chr21_predict_10kb.txt | sed s'/[:-]/\t/'g > chr21_predict_10kb.bedpe

## 转为hic
cat chr21_predict_10kb.bedpe | awk '{print 0,$1,$2,0,0,$4,$5,1,$7}' >  chr21_predict_10kb_tmp.txt
less chr21_predict_10kb_tmp.txt |awk '{OFS="\t"; for (i=0; i<$NF; i++) print }' |cut -f1-8 -d" " > chr21_predict_10kb_tmp2.txt
java -Xmx20g -jar ${juicer} pre -r 5000,10000,20000,25000,40000,50000,100000 chr21_predict_10kb_tmp2.txt chr18_predict_10kb.hic /data/public/refGenome/bwa_index/rheMac8/rheMac8.chrom.sizes

cat chr1_modify_hicplus_5kb.bedpe | awk '{print 0,$1,$2,0,0,$4,$5,1,$7}' >  chr1_modify_hicplus_5kb_tmp.txt
less chr1_modify_hicplus_5kb_tmp.txt |awk '{OFS="\t"; for (i=0; i<$NF; i++) print }' |cut -f1-8 -d" " > chr1_modify_hicplus_5kb_tmp2.txt
java -Xmx20g -jar ${juicer} pre -r 5000,10000,20000,25000,40000,50000,100000 chr1_modify_hicplus_5kb_tmp.txt chr1_modify_hicplus_5kb.hic hg38

cat chr1_modify_raw_5kb.bedpe | awk '{print 0,$1,$2,0,0,$4,$5,1,$7}' >  chr1_modify_raw_5kb_tmp.txt
less chr1_modify_raw_5kb_tmp.txt |awk '{OFS="\t"; for (i=0; i<$NF; i++) print }' |cut -f1-8 -d" " > chr1_modify_raw_5kb_tmp2.txt
java -Xmx20g -jar ${juicer} pre -r 5000,10000,20000,25000,40000,50000,100000 chr1_modify_raw_5kb_tmp.txt chr1_modify_raw_5kb.hic hg38




cat chr1_100_predict_5Kb.txt | sed s'/[:-]/\t/'g > chr1_100_predict_5Kb.bedpe
cat chr1_100_predict_5Kb.bedpe | awk '{print 0,$1,$2,0,0,$4,$5,1,$7}' >  chr1_100_predict_5Kb_tmp.txt
less chr1_100_predict_5Kb_tmp.txt |awk '{OFS="\t"; for (i=0; i<$NF; i++) print }' |cut -f1-8 -d" " > chr1_100_predict_5Kb_tmp2.txt
java -Xmx20g -jar ${juicer} pre -r 5000,10000,20000,25000,40000,50000,100000 chr1_100_predict_5Kb_tmp2.txt chr1_100_predict_5Kb.hic hg38
