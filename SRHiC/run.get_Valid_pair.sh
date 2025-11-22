#!/bin/bash

## raw counts
zcat /data/tang/cohesin_project/ChIA_PET_FINAL/Final.for.Downstream.Analysis/raw.data/GM12878_WT_ChIAPET_SMC1A.intra_iPET_ALL.bedpe.gz | \
	awk '$1==$4 && $5-$3>8000 && $5-$3<5000000{print }' | \
	perl -lane '{if($F[-2] eq "+"){$head_str=0} elsif($F[-2] eq "-"){$head_str=16}; if($F[-1] eq "+"){$tail_str=0} elsif($F[-1] eq "-"){$tail_str=16}; $head_cen=$F[1]+int(($F[2]-$F[1])/2); $tail_cen=$F[4]+int(($F[5]-$F[4])/2); print join("\t", ($head_str, $F[0], $head_cen, 0, $tail_str, $F[3], $tail_cen, 1)); }' | \
	perl -lane '{$s=$_; $s =~ s/chr//g; print $s}' | \
	LANG=C sort -k2,2d -k6,6d \
	    > GM12878_WT_ChIAPET_SMC1A.remove_Diagonal.txt

zcat /data/tang/cohesin_project/ChIA_PET_FINAL/Final.for.Downstream.Analysis/raw.data/GM12878_WT_ChIAPET_CTCF.intra_iPET_ALL.bedpe.gz | \
	perl -lane '{if($F[-2] eq "+"){$head_str=0} elsif($F[-2] eq "-"){$head_str=16}; if($F[-1] eq "+"){$tail_str=0} elsif($F[-1] eq "-"){$tail_str=16}; $head_cen=$F[1]+int(($F[2]-$F[1])/2); $tail_cen=$F[4]+int(($F[5]-$F[4])/2); print join("\t", ($head_str, $F[0], $head_cen, 0, $tail_str, $F[3], $tail_cen, 1)); }' | \
	perl -lane '{$s=$_; $s =~ s/chr//g; print $s}' | \
	LANG=C sort -k2,2d -k6,6d \
	    > GM12878_WT_ChIAPET_CTCF.intra_iPET_ALL.txt

zcat /data/yuhan/chiapet/hg38/9_GW11B3_d20_ChIAPET_CTCF_merged_B1_2_3/9_GW11B3_d20_ChIAPET_CTCF_merged_B1_2_3.linker.nodup.bedpe.gz | \
	perl -lane '{if($F[-2] eq "+"){$head_str=0} elsif($F[-2] eq "-"){$head_str=16}; if($F[-1] eq "+"){$tail_str=0} elsif($F[-1] eq "-"){$tail_str=16}; $head_cen=$F[1]+int(($F[2]-$F[1])/2); $tail_cen=$F[4]+int(($F[5]-$F[4])/2); print join("\t", ($head_str, $F[0], $head_cen, 0, $tail_str, $F[3], $tail_cen, 1)); }' | \
	perl -lane '{$s=$_; $s =~ s/chr//g; print $s}' | \
	LANG=C sort -k2,2d -k6,6d \
	    > 9_GW11B3_d20_ChIAPET_CTCF.intra_iPET_ALL.txt

## HICDC signal
cat /data/tang/cohesin_project/ChIA_PET_FINAL/Final.for.Downstream.Analysis/run.HiCDCPlus.each.sample/GM12878_WT_ChIAPET_SMC1A/GM12878_WT_ChIAPET_SMC1A.intra_iPET_ALL.qvalue.nb_hurdle.5kb.filter_0.05.bedpe | \
    perl -lane '{$head_cen=$F[1]+int(($F[2]-$F[1])/2); $tail_cen=$F[4]+int(($F[5]-$F[4])/2); print join("\t", (0, $F[0], $head_cen, 0, 0, $F[3], $tail_cen, 1));}' | \
	perl -lane '{$s=$_; $s =~ s/chr//g; print $s}' | \
	LANG=C sort -k2,2d -k6,6d \
	    > GM12878_WT_ChIAPET_SMC1A.intra_iPET_signal.txt
