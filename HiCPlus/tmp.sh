#!/bin/bash

cat DEstripe.txt | grep 'WT>SA1KO' | cut -f1 | sed s'/_/\t/'g | sort -k1,1 -k2,2n | awk 'BEGIN{OFS="\t"}{print $1,$2,$3,$4,$5,$6,1}' > Stripe.gain.WT_vs_SA1KO.bedpe
cat DEstripe.txt | grep 'WT<SA1KO' | cut -f1 | sed s'/_/\t/'g | sort -k1,1 -k2,2n | awk 'BEGIN{OFS="\t"}{print $1,$2,$3,$4,$5,$6,1}' > Stripe.loss.WT_vs_SA1KO.bedpe
cat DEstripe.txt | grep 'WT>SA2KO' | cut -f1 | sed s'/_/\t/'g | sort -k1,1 -k2,2n | awk 'BEGIN{OFS="\t"}{print $1,$2,$3,$4,$5,$6,1}' > Stripe.gain.WT_vs_SA2KO.bedpe
cat DEstripe.txt | grep 'WT<SA2KO' | cut -f1 | sed s'/_/\t/'g | sort -k1,1 -k2,2n | awk 'BEGIN{OFS="\t"}{print $1,$2,$3,$4,$5,$6,1}' > Stripe.loss.WT_vs_SA2KO.bedpe
cat DEstripe.txt | grep 'SA1KO>SA2KO' | cut -f1 | sed s'/_/\t/'g | sort -k1,1 -k2,2n | awk 'BEGIN{OFS="\t"}{print $1,$2,$3,$4,$5,$6,1}' > Stripe.gain.SA1KO_vs_SA2KO.bedpe
cat DEstripe.txt | grep 'SA1KO<SA2KO' | cut -f1 | sed s'/_/\t/'g | sort -k1,1 -k2,2n | awk 'BEGIN{OFS="\t"}{print $1,$2,$3,$4,$5,$6,1}' > Stripe.loss.SA1KO_vs_SA2KO.bedpe
