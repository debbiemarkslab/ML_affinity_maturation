#!/bin/bash
#SBATCH -c 1 
#SBATCH -n 1
#SBATCH -t 0-3:0:0
#SBATCH -p short
#SBATCH --mem=200GB
#SBATCH -o logs/generating_numbers.out
#SBATCH -e logs/generating_numbers.err
#SBATCH --mail-user=steffanpaul@g.harvard.edu
#SBATCH --mail-type=ALL

#Run this file after deduplicating and compiling all the sequences!!

#source this environment before running the job
source activate /n/groups/marks/users/phil/anaconda2_tf

#all rounds compile
cd bcsplit_protein

ANARCI -i NbB7FACS11_correct_nb_dedup2.fasta -o NbB7FACS11_correct_nb_dedup2 -s i --csv

