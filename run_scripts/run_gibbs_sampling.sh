#!/bin/bash

#source activate /home/stp022/.conda/envs/baff

python ML_affinity_maturation/scripts/gibbs_supervised.py \
        --ckptfnm models/AT110/CNN/Train_FACS1_MACS_enriched/ckpt \
        --numsteps 5 \
        --numseqs 10 \
        --temp_set 0 \
        --modeltype CNN