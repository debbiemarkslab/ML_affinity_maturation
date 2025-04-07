#!/bin/python

'''
Script for producing the input file needed for scoring
mutants with ESM LLM. Input options/modes:
1) --mode singles (Single mutant scoring):
    - wt_name (name of WT sequence used)
    - wt_align_fnm (path to the alignment of the WT sequence)
    - outfnm (the path to save the mutation file)
    
2) --mode dfs (X_df, Y_df input from affinity_maturation_utils):
    - 
    
3) --fasta (input a fasta file): ##IN DEV
    - wt_seq (the full WT sequence)
'''

import os, sys
sys.path.append('/n/groups/marks/projects/binding_affinity')
import affinity_maturation_utilities as util
from modelling_functions import *
from argparse import ArgumentParser
import joblib
import pandas as pd
from Bio.SeqIO.FastaIO import SimpleFastaParser


def write_fasta(seqs, ids, fnm):
    with open(fnm, 'w') as f:
        for (i,s) in zip(ids, seqs):
            f.write(f'>{i}\n')
            f.write(f'{s}\n')


def make_singles_mut_file(wt_name, wt_align_fnm, outfnm, withWT=True,
                         output_mode='11'):
    
    print (f'Making single mutants file name for {wt_name}')
    print (f'Saving at {outfnm}')
    
    mut_seqs, wt_seq, imgt_col = util.make_singles_muts(wt_align_fnm, wt_name, with_WT=withWT)

    #extract the mutation name
    mut_df = pd.DataFrame(mut_seqs.index, columns=['full_mut_name'])
    mut_df['mutant'] = mut_df['full_mut_name'].str.split('_', expand=True).iloc[:,1]
    mut_df['mutated_sequence'] = mut_seqs.values

    #save
    outdir = os.path.dirname(outfnm)
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    if output_mode[0] == '1':
        #csv save
        mut_df.to_csv(outfnm, index=False)
        
    if output_mode[1] == '1':
        #fasta save
        fasta_fnm = os.path.splitext(outfnm)[0] + '.fasta'
        write_fasta(mut_df['mutated_sequence'], mut_df['full_mut_name'], fasta_fnm)
    
    
def make_dfs_mut_file(X_fnm, Y_fnm, wt_align_fnm,
                      round_enriched, outfnm,
                     output_mode='11'):
    
    print ('Converting the seqs from')
    print (X_fnm)
    print (f'Found in the {round_enriched} data')
    print ('to an ESM mut file format...')
    print ('Saving to:')
    print (outfnm)

    X_df = pd.read_csv(X_fnm).set_index('seq_ID')
    Y_df = pd.read_csv(Y_fnm).set_index('seq_ID')

    #sub out the data
    val_df = Y_df.loc[~Y_df[f'{round_enriched}_split'].isna(), [f'{round_enriched}_logratio', f'{round_enriched}_enriched']]
    val_df['FullSeq_nogaps'] = X_df.loc[val_df.index, 'FullSeq_nogaps']

    #optionally only downselect to seqs that are the same length
    samelength = True
    if samelength:
        slidx = val_df['FullSeq_nogaps'].apply(len) == np.max(val_df['FullSeq_nogaps'].apply(len))
        val_df = val_df[slidx.values]

    #label the sequences by their mutations from WT
    wt_df, imgt_cols = util.extract_cdrs(wt_align_fnm, **{'return_col_labels':True})
    wt_seq = wt_df.loc[0, 'FullSeq_nogaps']

    val_df['mut_names'] = val_df['FullSeq_nogaps'].apply(lambda s: util.id_mut_from_seq(wt_seq, s, justnames=True))
    val_df['num_muts'] = val_df['mut_names'].apply(len)
    val_df['mutant'] = val_df['mut_names'].apply(lambda s: ':'.join(s))
    val_df = val_df.rename({'FullSeq_nogaps':'mutated_sequence'}, axis=1)
    #val_df['full_mut_name'] = ['seq_' + str(s) for s in range(val_df.shape[0])]
    val_df['full_mut_name'] = pd.Series(val_df.index).str.replace('\/' , '_').values

    #add in a WT name for any empty mutants
    val_df.loc[val_df['mutant'] == '', 'mutant'] = 'Q1Q'
    
    #save
    outdir = os.path.dirname(outfnm)
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    if output_mode[0] == '1':
        #csv save
        csv_fnm = os.path.splitext(outfnm)[0] + '.csv'
        val_df.to_csv(csv_fnm, index=False)
        
    if output_mode[1] == '1':
        #fasta save
        fasta_fnm = os.path.splitext(outfnm)[0] + '.fasta'
        write_fasta(val_df['mutated_sequence'], val_df['full_mut_name'], fasta_fnm)
        

    
if __name__ == '__main__':
    
    parser = ArgumentParser()
    parser.add_argument('--mode', type=str, required=True,
                       help='One of <singles>, <dfs>')
    
    #general arguments
    parser.add_argument('--wt_align_fnm')
    parser.add_argument('--outfnm')
    
    #singles arguments
    parser.add_argument('--wt_name')
    parser.add_argument('--withWT', action='store_true', default=True)
    
    #dfs arguments
    parser.add_argument('--X_fnm')
    parser.add_argument('--Y_fnm')
    parser.add_argument('--round_enriched')
    
    args = parser.parse_args()
    
    if args.mode == 'singles':
        make_singles_mut_file(args.wt_name, args.wt_align_fnm,
                              args.outfnm, args.withWT)
        
    if args.mode == 'dfs':
        make_dfs_mut_file(args.X_fnm, args.Y_fnm, args.wt_align_fnm,
                          args.round_enriched, args.outfnm)
                          
        
    
    