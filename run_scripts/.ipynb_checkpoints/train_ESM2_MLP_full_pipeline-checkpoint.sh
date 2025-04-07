#!/bin/bash
#SBATCH -c 1                               # Request one core
#SBATCH -p short                           # Partition to run in
#SBATCH -t 0-3:00
#SBATCH --mem=16G                         # Memory total in MiB (for all cores)
#SBATCH --mail-user steffanpaul@g.harvard.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name="ESM2_MLP_pipeline"
#SBATCH -o logs/%x_%j_%u.out
#SBATCH -e logs/%x_%j_%u.err


module load gcc/9.2.0 cuda/11.7 
source activate baff

#run from repo dir

###########################################################

#PARAMS TO SET
WT_NAME=$1
DATA_DIR_NM=$2
DATA_DIR_MID=$3


MAKE_MUT_FILE=true
EXTRACT_REPS=true
TRAIN_MODEL=true

###########################################################

#PARAMS WITH INHERITED VALUES
if [ $WT_NAME = "AT110" ] ; then
    DATA_DIR_NM="AT1R_affinity" 
    DATA_DIR_MID="V1"
fi
if [ $WT_NAME = "NbB7" ] ; then
    DATA_DIR_NM="B2AR_affinity" 
    DATA_DIR_MID="V1"
fi
if [ $WT_NAME = "LysoNbV1" ] ; then
    DATA_DIR_NM="TMEM192_affinity" 
    DATA_DIR_MID="V1"
fi
if [ $WT_NAME = "RX002" ] ; then
    DATA_DIR_NM="RXFP1_affinity" 
    DATA_DIR_MID="V1"
fi

#sequencing data paths
WT_ALIGN_FNM=data/${WT_NAME}/aligned_WT_sequence.csv
X_FNM=data/${WT_NAME}/X.csv
Y_FNM=data/${WT_NAME}/Y.csv

#########################################################

# MAKE MUT FILES

if [ $MAKE_MUT_FILE = true ] ; then
for ROUND in FACS1_MACS FACS2_MACS
do
OUTFNM=trained_models/${WT_NAME}/compmodels/ESM2_MLP/${ROUND}_seqs_mut_file.csv
python scripts/make_mutant_file.py \
        --mode dfs \
        --X_fnm $X_FNM \
        --Y_fnm $Y_FNM \
        --wt_align_fnm $WT_ALIGN_FNM \
        --outfnm $OUTFNM \
        --round_enriched ${ROUND} 
done     
fi


#########################################################

# EXTRACT REPS

if [ $EXTRACT_REPS = true ] ; then
source deactivate
source activate esm
OUT_PRE=trained_models/${WT_NAME}/compmodels_ESM2_MLP/
#singles
echo "Extracting reps for singles"
python scripts/esm_extract_embeddings.py \
            esm2_t33_650M_UR50D \
            ${OUT_PRE}/${WT_NAME}/${WT_NAME}_singles_mut_file.fasta \
            ${OUT_PRE}/${WT_NAME}/singles \
            --repr_layers 0 32 33 --include mean per_tok
               
#F1/M seqs
echo "Extracting reps for FACS1/MACS"
python scripts/esm_extract_embeddings.py \
            esm2_t33_650M_UR50D \
            ${OUT_PRE}/${WT_NAME}/FACS1_MACS_seqs_mut_file.fasta \
            ${OUT_PRE}/${WT_NAME}/F1_M_seqs \
            --repr_layers 0 32 33 --include mean per_tok
            
#F2/M seqs
echo "Extracting reps for FACS2/MACS"
python scripts/esm_extract_embeddings.py \
            esm2_t33_650M_UR50D \
            ${OUT_PRE}/${WT_NAME}/FACS2_MACS_seqs_mut_file.fasta \
            ${OUT_PRE}/${WT_NAME}/F2_M_seqs \
            --repr_layers 0 32 33 --include mean per_tok
conda deactivate
fi


#########################################################

# TRAIN MODEL
if [ $TRAIN_MODEL = true ] ; then
source activate baff
OUT_PRE=trained_models/${WT_NAME}/compmodels_ESM2_MLP/
for Y in logratio enriched
do
python  scripts/MLP_topmodel.py\
    --proj ${WT_NAME} \
    --X_fnm ${DATA_DIR}/${DATA_DIR_NM}/${DATA_DIR_MID}/X.csv \
    --Y_fnm ${DATA_DIR}/${DATA_DIR_NM}/${DATA_DIR_MID}/Y.csv \
    --train_seqs F1_M_seqs \
    --test_seqs F2_M_seqs \
    --train_Y_col FACS1_MACS_${Y} \
    --test_Y_col FACS2_MACS_${Y} \
    --ckpt_dir ${OUT_PRE}/${WT_NAME}/F1_M_seqs_ckpts/MLP_topmodel_${Y} \
    --num_epochs 200
done

fi