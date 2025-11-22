#!/bin/bash

#####################################################################################################################################################################################
## human
## input
input_bedpe=/data/yuhan/chiapet/hg38/9_GW11B3_d20_ChIAPET_CTCF_merged_B2_3/9_GW11B3_d20_ChIAPET_CTCF_merged_B2_3.linker.nodup.bedpe.gz
## tmp file
tmp_hic_txt=9_GW11B3_d20_ChIAPET_CTCF_merged_B2_3.hic.format.txt
## output file
output_hic=9_GW11B3_d20_ChIAPET_CTCF_merged_B2_3.remove_diagonal.hic

zcat ${input_bedpe} | \
    perl -lane 'if($F[0] eq $F[3] && $F[5]-$F[1] > 8000 && $F[5]-$F[1] <= 2000000){print}' | \
    perl -lane '{if($F[-2] eq "+"){$head_str=0} elsif($F[-2] eq "-"){$head_str=16}; if($F[-1] eq "+"){$tail_str=0} elsif($F[-1] eq "-"){$tail_str=16}; $head_cen=$F[1]+int(($F[2]-$F[1])/2); $tail_cen=$F[4]+int(($F[5]-$F[4])/2); print join("\t", ($head_str, $F[0], $head_cen, 0, $tail_str, $F[3], $tail_cen, 1)); }' | \
    perl -lane '{$s=$_; $s =~ s/chr//g; print $s}' | \
    LANG=C sort -k2,2d -k6,6d > ${tmp_hic_txt}

java -Xmx20g -jar ${juicer} pre -r 1000000,500000,100000,50000,25000,10000,5000,1000 \
	 ${tmp_hic_txt} \
	 ${output_hic} \
	 /data2/yuhan/Project/Cohesin/differentialLoops/HiCDCPlus/hicFile/chrSize.txt
rm ${tmp_hic_txt}


#####################################################################################################################################################################################
## macaque
## input
input_bedpe=/data/yuhan/chiapet/rheMac8/RheNeu_d12_ChIAPET_CTCF_merged_B1_2/RheNeu_d12_ChIAPET_CTCF_merged_B1_2.linker.nodup.bedpe.gz
## tmp file
tmp_hic_txt=RheNeu_d12_ChIAPET_CTCF_merged_B1_2.hic.format.txt
## output file
output_hic=RheNeu_d12_ChIAPET_CTCF_merged_B1_2.remove_diagonal_10K.hic

zcat ${input_bedpe} | \
    perl -lane 'if($F[0] eq $F[3] && $F[5]-$F[1] > 8000){print}' | \
    perl -lane '{if($F[-2] eq "+"){$head_str=0} elsif($F[-2] eq "-"){$head_str=16}; if($F[-1] eq "+"){$tail_str=0} elsif($F[-1] eq "-"){$tail_str=16}; $head_cen=$F[1]+int(($F[2]-$F[1])/2); $tail_cen=$F[4]+int(($F[5]-$F[4])/2); print join("\t", ($head_str, $F[0], $head_cen, 0, $tail_str, $F[3], $tail_cen, 1)); }' | \
    perl -lane '{$s=$_; $s =~ s/chr//g; print $s}' | \
    LANG=C sort -k2,2d -k6,6d > ${tmp_hic_txt}

# 由于进行normalization时报错，因此添加-n参数，不进行normalization
java -Xmx20g -jar ${juicer} pre -r 10000 -n \
	 ${tmp_hic_txt} \
	 ${output_hic} \
	 /data/public/refGenome/bwa_index/rheMac8/rheMac8.chrom.sizes
rm ${tmp_hic_txt}
