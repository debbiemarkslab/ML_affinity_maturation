#!/bin/bash
#SBATCH -c 1                               # Request one core
#SBATCH -t 0-02:00                         # Runtime in D-HH:MM format
#SBATCH -p short                           # Partition to run in
#SBATCH --mem=16G                         # Memory total in MiB (for all cores)
#SBATCH -o logs/train_LR_%j.out                 
#SBATCH -e logs/train_LR_%j.err                 
#SBATCH --mail-user steffanpaul@g.harvard.edu
#SBATCH --mail-type=ALL

source activate /home/stp022/.conda/envs/baff

#move back to repo dir
cd ../


seqtype=FullSeqs
gaptype=withgaps

data_fol=AT110/V1
data_path=data/${data_fol}

exp_dir=trained_models/
experiment=AT110/
sub_experiment=compmodels/linear/

for train in FACS1_MACS_logratio FACS2_MACS_logratio
do

test=FACS2_MACS_logratio

########################################################################################


MODEL_NM=${experiment}:${sub_experiment}:Train_${train}

echo ${MODEL_NM}
### TRAIN


python scripts/train_LR_scikit.py \
        --model linear_regression \
        --data_dir ${data_path} \
        --data_sub ${data_sub} \
        --exp_dir ${exp_dir} \
        --experiment_name ${experiment} \
        --sub_experiment ${sub_experiment} \
        --CDR_choice FullSeqs_withgaps \
        --gap_cutoff 1.0 \
        --no_wt_col \
        --Y_col ${train} \
        --l1_lambda 5e-4 \
        --split train

### TEST

python scripts/test_LR_scikit.py \
            --model linear_regression \
            --data_dir ${data_path} \
            --data_sub ${data_sub} \
            --exp_dir ${exp_dir} \
            --experiment_name ${experiment} \
            --sub_experiment ${sub_experiment}  \
            --CDR_choice FullSeqs_withgaps \
            --gap_cutoff 1.0 \
            --no_wt_col \
            --train_Y_col ${train} \
            --Y_col ${test} \
            --split test 

        