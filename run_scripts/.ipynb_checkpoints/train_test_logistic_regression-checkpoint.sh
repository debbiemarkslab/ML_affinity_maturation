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

## SET PARAMETERS
data_fol=AT110/V1 #Where the data is located
data_path=data/${data_fol}

exp_dir=trained_models/
experiment=AT110/
sub_experiment=compmodels/logistic/
## 

for train in FACS1_MACS_enriched FACS2_MACS_enriched
do

test=FACS2_MACS_enriched

########################################################################################


MODEL_NM=${experiment}:${sub_experiment}:Train_${train}

echo ${MODEL_NM}
### TRAIN


python scripts/train_LR_scikit.py \
        --model logistic_regression \
        --data_dir ${data_path} \
        --data_sub ${data_sub} \
        --exp_dir ${exp_dir} \
        --experiment_name ${experiment} \
        --sub_experiment ${sub_experiment} \
        --CDR_choice FullSeqs_withgaps \
        --gap_cutoff 1.0 \
        --no_wt_col \
        --Y_col ${train} \
        --l1_lambda 2 \
        --split train

### TEST

python scripts/test_LR_scikit.py \
            --model logistic_regression \
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

        