#!/bin/bash
#SBATCH -c 1                               # Required number of CPUs
#SBATCH -n 1
#SBATCH -t 0-5:00                         # Runtime in D-HH:MM format
#SBATCH -p short  # Adding more partitions for quick testing                         # Partition to run in
#SBATCH --mem=16G                         # Memory total in MiB (for all cores)
#SBATCH --mail-user steffanpaul@g.harvard.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name="AT110_train_NN"

#Job array specific
#SBATCH -o logs/%x_%j_%u.out
#SBATCH -e logs/%x_%j_%u.err


module load gcc/6.2.0 cuda/11.2
source activate baff

MODEL=CNN

## SET PARAMETERS
DATA_DIR=data/AT110
EXP_DIR=trained_models
EXPERIMENT_NAME=AT110
SUB_EXPERIMENT=compmodels/${MODEL}


for METRIC in enriched logratio
do
    for TRAIN_Y_COL in FACS1_MACS FACS2_MACS
    do
        #inherits values
        Y_COL=${TRAIN_Y_COL}_${METRIC}
        TEST_Y_COL=FACS2_MACS_${METRIC}

        #train
        python /n/groups/marks/projects/binding_affinity/model_zoo/train_supervised_model.py \
                --model ${MODEL} \
                --data_dir ${DATA_DIR} \
                --data_sub manual \
                --exp_dir ${EXP_DIR} \
                --experiment_name ${EXPERIMENT_NAME} \
                --sub_experiment ${SUB_EXPERIMENT} \
                --CDR_choice FullSeqs_withgaps \
                --Y_col ${Y_COL} \
                --l1_lambda 0.0001 \
                --num_epochs 1000
    done
done
        

        




