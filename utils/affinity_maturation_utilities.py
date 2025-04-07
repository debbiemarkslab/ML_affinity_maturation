import os, sys, re
import joblib
import time as time
import itertools
import pandas as pd
import numpy as np
import scipy.special
import numbers
from Bio.SeqIO.FastaIO import SimpleFastaParser
from functools import reduce
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
from itertools import combinations, chain
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
sys.path.append('/n/groups/marks/projects/binding_affinity')
from modelling_functions import *

import warnings
warnings.filterwarnings("ignore")


### FUNCTIONS FOR PROCESSING THE SEQUENCES IN PREP FOR ANALYSIS

def import_fasta(fnm, output='df'):
    '''
    Simple function for importing a fasta as a df indexed by the id
    '''
    seq_info = {}
    with open(fnm) as fasta_file:
        for title, seq in SimpleFastaParser(fasta_file):
            seq_info[title] = seq
    #convert to dict and sort and then back to dict
    if output == 'df':
        seq_df = pd.DataFrame.from_dict(seq_info, orient='index', columns=['seq'])
        return (seq_df)
    elif output == 'dict':
        return (seq_info)
    
    
def write_fasta(ids, seqs, outfnm):
    '''
    Simple function for writing a fasta
    '''
    with open(outfnm, 'w') as f:
        for i,s in zip(ids, seqs):
            f.write(f'>{i}\n')
            f.write(f'{s}\n')
        

def deduplicate(DIR, fname, justcheck=False):
    '''
    Deduplicates a fasta file and aggregates the counts
    
    
    Inputs:
    
        DIR: t–he directory in which the fasta file recides
        fname: the name of the fasta file without the .fasta extension
        justcheck: if True, doesn't save the deduplicated fasta, just reportsx
        
    Outputs:
    ii
        Writes a fasta file with all the counts from duplicate sequences aggregated
    '''
    infnm = os.path.join(DIR, f'{fname}.fasta')
    outfnm = os.path.join(DIR, f'{fname}_dedup.fasta')
    #read in sequences and track duplicates
    seq_info = {}
    num_dup_seqs = 0
    with open(infnm) as fasta_file:
        for title, seq in SimpleFastaParser(fasta_file):
            count = int(title.split("-")[-1])
            #track duplicates
            if seq in seq_info.keys():
                seq_info[seq] += count
                num_dup_seqs += 1
            else:
                #add new seq to dict
                seq_info[seq] = count
    #convert to dict and sort and then back to dict
    seq_df = pd.DataFrame.from_dict(seq_info, orient='index', columns=['counts'])
    seq_df = seq_df.sort_values('counts', ascending=False)
    seq_dict = seq_df.to_dict(orient='dict')['counts']
    #report number of dulicates
    print (f'{fname}')
    print(f'{num_dup_seqs} duplicates found')
    #save
    if not justcheck:
        print ('writing deduplicated fasta...')
        with open(outfnm, 'w') as f:
            for ii,(seq, cnt) in enumerate(seq_dict.items()):
                f.write(f'>{ii+1}-{cnt}\n')
                f.write(f'{seq}\n')
                

def length_filter_protein(DIR, fname, lenrange, printcheck=False):
    '''
    Filters out unusually short sequences
    
    
    Inputs:
    
        DIR: the directory in which the fasta file recides
        fname: the name of the fasta file without the .fasta extension
        lenrange: the lower and upper length range (inclusive of both)
        justcheck: if True, just plots the length distribution, doesn't filter anything
        
    Outputs:
    ii
        Writes a fasta file with all the short sequences removed
    '''
    print (fname)
    infnm = os.path.join(DIR, f'{fname}.fasta')
    outfnm = os.path.join(DIR, f'{fname}_lenfil.fasta')
    #read in sequences and track duplicates
    seq_info = {}
    num_dup_seqs = 0
    with open(infnm) as fasta_file:
        for title, seq in SimpleFastaParser(fasta_file):
            count = int(title.split("-")[-1])
            #track duplicates
            if seq in seq_info.keys():
                seq_info[seq] += count
                num_dup_seqs += 1
            else:
                #add new seq to dict
                seq_info[seq] = count
    #convert to dict and sort and then back to dict
    seq_df = pd.DataFrame.from_dict(seq_info, orient='index', columns=['counts'])
    seq_df = seq_df.sort_values('counts', ascending=False)
    seq_df = seq_df.reset_index().rename({'index':'seq'}, axis=1)
    seq_df['length'] = seq_df['seq'].apply(len)
    if printcheck:
        print (seq_df['length'].value_counts().sort_index())
    seq_df2 = seq_df[(seq_df['length'] >= lenrange[0]) & (seq_df['length'] <= lenrange[1])]
    print ('Number of sequences before filtering')
    print (seq_df.shape[0])                     
    print ('Number of sequences after filtering')
    print (seq_df2.shape[0]) 
    seq_df2.index = seq_df2['seq']
    seq_dict = seq_df2.to_dict(orient='dict')['counts']
    if not printcheck:
        print ('writing len filtered fasta...')
        with open(outfnm, 'w') as f:
            for ii,(seq, cnt) in enumerate(seq_dict.items()):
                f.write(f'>{ii+1}-{cnt}\n')
                f.write(f'{seq}\n')
            
            
            
def get_round_info_from_fasta(fname, seq_round):
    '''
    Opens a fasta file. The fasta file should be formatted
    such that the name of each sequence contains the id of the sequence
    followed by the counts from the sequencing round.
    
    Inputs:
    
        fname: filename of the fasta file
        seq_round: the name of the sequencing round
        
    Outputs:
    
        r_info: a dataframe indexed by the sequence with a column for the count
    '''
    r_info = []
    seq_info = {}
    dup_seqs = []
    print(fname)
    with open(fname) as fasta_file:
        for title, seq in SimpleFastaParser(fasta_file):
            count = int(title.split("-")[-1])
            if seq in seq_info.keys():  # no duplicate sequences allowed
                raise Exception("At {}, found second instance of sequence {}".format(title, seq))
            else:
                seq_info[seq] = {"{}_ct".format(seq_round): count, 
                             "{}_id".format(seq_round): title}
    seq_info = pd.DataFrame.from_dict(seq_info, orient='index')
    seq_info.index.name = 'seq'
    r_info.append(seq_info)
    r_info = reduce(lambda x, y: pd.merge(x, y, how='outer', left_index=True, right_index=True), r_info)
    return r_info


def compile_fastas(label_dict, outdir, campaign_name):
    '''
    Compiled all fasta files from a campaign into a single fasta
    with all unique sequences from a campaign. This can then be used
    for alignment
    
    
    Inputs:
    
        label_dict: a dictionary keyed by the label of the round, and the value being the fnm
        outdir: the directory where the compiled fasta will be written
        campaign_name: the name of this campaign. The name of the sort rounds will be appended to the 
                        name so you know what data we have for each sequence
    '''
    #extract the sequences from all the rounds
    r_info = []
    for seq_round, fname in label_dict.items():
        seq_info = {}
        #parse fasta file
        with open(fname) as fasta_file:
            for title, seq in SimpleFastaParser(fasta_file):
                count = int(title.split("-")[-1])
                if seq in seq_info.keys():  # no duplicate sequences allowed
                    raise Exception("At {}, found second instance of sequence {}".format(title, seq))
                seq_info[seq] = {"{}_ct".format(seq_round): count, 
                                 "{}_id".format(seq_round): title}
        seq_info = pd.DataFrame.from_dict(seq_info, orient='index')
        seq_info.index.name = 'seq'
        r_info.append(seq_info)
    r_info = reduce(lambda x, y: pd.merge(x, y, how='outer', left_index=True, right_index=True), r_info)
    final_round = list(label_dict.keys())[-1]
    r_info = r_info.sort_values(f'{final_round}_ct', ascending=False)
    
    #assign a new id
    r_info['new_id'] = np.arange(r_info.shape[0])
    #create a new fasta label for each sequence
    rounds = list(label_dict.keys())
    tmp = r_info[['new_id'] + [f'{r}_ct' for r in rounds]]
    tmp[tmp.isna()] = 0
    r_info['new_label'] = tmp.apply(lambda x: '-'.join([str(int(xx)) for xx in x]), axis=1)
    
    #save the sequences as a fasta file
    round_names = '_'.join(rounds)
    outfnm = os.path.join(outdir, f'{campaign_name}_{round_names}.fasta')
    print ('writing compiled fasta file...')
    with open(outfnm, 'w') as f:
        for seq,label in r_info['new_label'].to_dict().items():
            f.write(f'>{label}\n')
            f.write(f'{seq}\n')


def compile_rnd_alignments(round_dict, wt_afnm, save_fnm):
    '''
    Compile individual alignments corresponding to different sequencing
    rounds based on a reference alignment. Only IMGT columns present in the
    WT sequence will be included in the final alignment
    
    Inputs:
        round_dict: a dictionary keyed by the round name and with values for the fnm
                    to each individual alignment
        wt_afnm: the WT alignment to use as a reference
        save_fnm: where to save the resulting file
    '''
    
    #import reference
    a_df = pd.read_csv(wt_afnm)
    aa_df = a_df.iloc[:, a_df.columns.str.match('\d+')]
    aa_cols = aa_df.columns[aa_df.iloc[0] != '-']


    r_list = []
    #import rounds alignments
    for rlab,rfnm in tqdm(round_dict.items(), desc='compiling alignments'):

        r_df = pd.read_csv(rfnm)
        #extract the counts
        r_counts = r_df['Id'].str.extract('\d+-(\d+)', expand=False).astype(int)
        #sub out only the parts present in the reference
        r_seqs = r_df.loc[:,aa_cols].apply(''.join, axis=1)
        #deduplicate now that we have reduced the sequence length
        r_seqdf = pd.concat([r_counts, r_seqs], axis=1)
        r_seqdf.columns = [f'{rlab}_ct', 'seq']
        r_seqdedup = r_seqdf.groupby('seq').sum().sort_values(f'{rlab}_ct', ascending=False)
        r_list.append(r_seqdedup)

    r_info = pd.concat(r_list, axis=1).fillna(0).astype(int).astype(str)
    r_cols = r_info.columns
    r_info['idx'] = np.arange(r_info.shape[0]).astype(str)
    r_info = r_info[['idx'] + list(r_cols)]
    r_info['Id'] = r_info.apply('-'.join, axis=1)

    #put back as a matrix
    sdf = pd.Series(r_info.index).apply(list)
    sdf = pd.DataFrame(np.vstack(sdf.values))
    sdf.columns = aa_cols
    sdf['Id'] = r_info['Id'].values
    sdf = sdf[['Id'] + list(aa_cols)]
    
    #save
    #make sure the folder exists for the save_fnm
    save_dir = os.path.dirname(os.path.realpath(save_fnm))
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    sdf.to_csv(save_fnm, index=False)
    
    
def get_alignment_imgt_cols(a_df):
    a_df_imgt_cols = a_df.iloc[:, a_df.columns.str.match('\d+')]
    return (a_df_imgt_cols)

def extract_cdrs(file, **kwargs):
    '''
    Function that extracts the CDR sequences from an ANARCI output
    
    Input:
        
        file: ANARCI output file name
        gap_cutoff: the threshold to remove gapped columns
        return_col_labels: whether or not to return a dictionary containing the IMGT
                            indices of each residue in each column 
        
    Output:
    
        df: a dataframe with separate columns for the gapped and ungapped sequences
    '''
    
    if not 'gap_cutoff' in kwargs:
        kwargs['gap_cutoff'] = 0.2
    gap_cutoff = kwargs['gap_cutoff']
    
    if not 'return_col_labels' in kwargs:
        kwargs['return_col_labels'] = False
    return_col_labels = kwargs['return_col_labels']
    
    #also construct a dict containing the IMGT used within each gapped column
    imgt_col_labels = {}
    
    df = pd.read_csv(file)
    
    #Ungapped sequences
    df['FullSeq_nogaps'] = df.iloc[:, df.columns.str.match('\d+')].apply(
        lambda x: ''.join(x).replace('-',''), axis=1)
    df['FullSeqs_len'] = df['FullSeq_nogaps'].apply(len)
    df['CDR1_nogaps'] = df.loc[:,'27':'38'].apply(lambda x: ''.join(x).replace('-',''), axis=1)
    df['CDR1_len'] = df['CDR1_nogaps'].apply(len)
    df['CDR2_nogaps'] = df.loc[:,'55':'65'].apply(lambda x: ''.join(x).replace('-',''), axis=1)
    df['CDR2_len'] = df['CDR2_nogaps'].apply(len)
    df['CDR3_nogaps'] = df.loc[:,'105':'117'].apply(lambda x: ''.join(x).replace('-',''), axis=1)
    df['CDR3_len'] = df['CDR3_nogaps'].apply(len)
    df['CDRS_nogaps'] = df['CDR1_nogaps'] + df['CDR2_nogaps'] + df['CDR3_nogaps']
    
    #wth gaps
    df['FullSeqs_withgaps'] = df.iloc[:, df.columns.str.match('\d+')].apply(
        lambda x: ''.join(x), axis=1)
    imgt_col_labels['FullSeqs_withgaps'] = df.iloc[:, df.columns.str.match('\d+')].columns.values
    df['CDR1_withgaps'] = df.loc[:,'27':'38'].apply(lambda x: ''.join(x), axis=1)
    imgt_col_labels['CDR1_withgaps'] = df.loc[:,'27':'38'].columns.values
    df['CDR2_withgaps'] = df.loc[:,'55':'65'].apply(lambda x: ''.join(x), axis=1)
    imgt_col_labels['CDR2_withgaps'] = df.loc[:,'55':'65'].columns.values
    df['CDR3_withgaps'] = df.loc[:,'105':'117'].apply(lambda x: ''.join(x), axis=1)
    imgt_col_labels['CDR3_withgaps'] = df.loc[:,'105':'117'].columns.values
    df['CDRS_withgaps'] = df['CDR1_withgaps'] + df['CDR2_withgaps'] + df['CDR3_withgaps']
    imgt_col_labels['CDRS_withgaps'] = np.hstack([imgt_col_labels[f'CDR{c}_withgaps'] for c in [1,2,3]])
    
    #first get the columns to keep after trimming
    dfs = df.iloc[:, df.columns.str.match('\d+')]
    gap_freq = dfs.apply(lambda x: x=='-').mean(0)
    keep_cols = gap_freq.index[gap_freq < gap_cutoff]
    df_keep = df.loc[:, keep_cols]
    #trimmed gaps according to a threshold
    df['FullSeqs_trimgaps'] = df_keep.apply(lambda x: ''.join(x), axis=1)
    imgt_col_labels['FullSeqs_trimgaps'] = keep_cols.values
    df['CDR1_trimgaps'] = df_keep.loc[:,'27':'38'].apply(lambda x: ''.join(x), axis=1)
    imgt_col_labels['CDR1_trimgaps'] = df_keep.loc[:, '27':'38'].columns.values
    df['CDR2_trimgaps'] = df_keep.loc[:,'55':'65'].apply(lambda x: ''.join(x), axis=1)
    imgt_col_labels['CDR2_trimgaps'] = df_keep.loc[:, '55':'65'].columns.values
    df['CDR3_trimgaps'] = df_keep.loc[:,'105':'117'].apply(lambda x: ''.join(x), axis=1)
    imgt_col_labels['CDR3_trimgaps'] = df_keep.loc[:, '105':'117'].columns.values
    df['CDRS_trimgaps'] = df['CDR1_trimgaps'] + df['CDR2_trimgaps'] + df['CDR3_trimgaps']
    imgt_col_labels['CDRS_trimgaps'] = np.hstack([imgt_col_labels[f'CDR{c}_trimgaps'] for c in [1,2,3]])
    
    if return_col_labels:
        return (df, imgt_col_labels)
    else:
        return (df)
    
    
def import_from_alignment(align_fnm, save_dir, data_name,
                        round_labs, **kwargs):
    '''
    Upon aligning the sequences from a fasta, use this function to 
    extract the sequences and the counts information.
    It expands the ID column from the alignment. Condenses the
    alignment matrix into the useful subsequences and arranges a 
    unique sequence ID.
    
    Inputs:
        align_fnm: the file name to the ANARCI output
        save_dir: the directory where the file will be saved
                    (for saving the IMGT columns)
        data_name: the name of the data for making a seq_ID
        round_labs: the names of the rounds from which the counts 
                    come from (as ordered in the FASTA id)
        
    Outputs:
        X_df: a prepared sequence dataset
        Y_df: a prepared metrics dataset
        imgt_labels: the imgt_labels for each of the subsequences
                        in X_df
    '''
    if not 'return_col_labels' in kwargs:
        kwargs['return_col_labels'] = True
        
    if not 'gap_cutoff' in kwargs:
        kwargs['gap_cutoff'] = 0.2
        
    if not 'choose_subseq' in kwargs:
        kwargs['choose_subseq'] = ['FullSeq_nogaps', 'CDR3_nogaps',
                 'FullSeqs_withgaps', 'CDR1_withgaps', 'CDR2_withgaps', 'CDR3_withgaps', 'CDRS_withgaps',
                'FullSeqs_trimgaps', 'CDR1_trimgaps', 'CDR2_trimgaps', 'CDR3_trimgaps', 'CDRS_trimgaps']
        
    if not 'plot_gaps' in kwargs:
        kwargs['plot_gaps'] = False
        
    
    #import the sequences
    ST = time.time()
    a_df, imgt_labels = extract_cdrs(align_fnm, **kwargs)
    print ('Extracting alignment took: {:.2f}s'.format(time.time()-ST))
    #save the imgt labels
    joblib.dump(imgt_labels, os.path.join(save_dir, 'imgt_labels.sav'))
    #extract a cleaned version
    a_df.reset_index(inplace=True)
    X_df = a_df[kwargs['choose_subseq']]

    #extract counts form ID in a_df
    Y_df = a_df['Id'].str.split('-', expand=True).apply(pd.to_numeric)
    Y_df.columns = ['index'] + [f'{r}_ct' for r in round_labs]

    #use the index unsplit from the fasta ID as the index
    X_df['index'] = Y_df['index']
    #save a unique id for this dataset
    Y_df['seq_ID'] = data_name + '_' + Y_df['index'].astype(str)
    X_df['seq_ID'] = data_name + '_' + X_df['index'].astype(str)
    
    if kwargs['plot_gaps']:
        show_removed_cols(a_df, kwargs['gap_cutoff'])
    
    return (Y_df, X_df, imgt_labels)


def show_removed_cols(a_df, gap_cutoff):
    
    '''
    Sub function for showing which columns fell below the gap
    threshold
    '''
    
    #show which columns were rmoved in the gap trimming
    #which gaps are being trimmed?
    dfs = a_df.iloc[:, a_df.columns.str.match('\d+')]
    gap_freq = dfs.apply(lambda x: x=='-').mean(0)
    gap_cutoff = 0.2

    tick_mask = np.linspace(0,len(gap_freq)-1, 20).astype(int)


    plt.figure()
    sns.barplot(x = gap_freq.index, y=gap_freq, color='black')
    plt.xticks(ticks = np.arange(len(gap_freq))[tick_mask],
               labels = gap_freq.index[tick_mask],
              rotation=90)
    plt.axhline(y=gap_cutoff, color='r', linestyle='--')
    plt.xlabel('IMGT column')
    plt.ylabel('Gap Frequency')

    plt.show()

    print (gap_freq.index[gap_freq > gap_cutoff])
    


def binarize_enrichment(Y_df, col, assignment_method='GMM',
                        plot_check = True, GMM_tr = 0.8, assignment_tr=(None,None),
                        only_ovlp = True, split_data=True, test_size=0.2, new_norm=False,
                       plot_kwargs={}):

    '''
    Take logratio values and binarize them to 0 or 1 enrichment
    Assignment can either be done with a Gaussian Mixture Model or 
    manually assigning thresholds. The function adds a new column
    to an existing df with the binarized value
    
    Inputs:
        Y_df: the df with the phenotypes (must have at least logratios and hasreads)
        assignment_method: [GMM or manual or sigma]
        plot_check: whether to plot the distribution of assigned values
        
        GMM_tr: the confidence threshold for assigning seqs to a 0 or 1
        assignment_tr: the manual thresholds. A tuple with (low, high) for 
                        separately considering low confidence assignments
                        
        only_ovlp: whether to include only the seqs present in both round
                    if False uses the imputed values for non-overalp
        split_data: whether to add a data split column for each variable in the dataset
    
    Outputs:
        Y_df: updated df with added columns
    
    '''
    #init some plotting arguments
    if 'palette' not in plot_kwargs:
        plot_kwargs['palette'] = 'RdBu'
    
    
    #init an enrichment columns
    r1,r0 = col.split('_')
    Y_df[f'{col}_enriched'] = np.nan
    
    #include seqs not in overlap or not
    if only_ovlp:
        has_idx = Y_df[f'{r1}_hasread'] & Y_df[f'{r0}_hasread']
    else:
        if new_norm:
            #only keep seqs seen in at least one round
            has_idx = Y_df[f'{r1}_hasread'] | Y_df[f'{r0}_hasread']
        else:
            has_idx = pd.Series(np.ones(Y_df.shape[0]).astype(bool))
    #logratio of 0 set as NA by default
    nz_idx = (Y_df[f'{col}_logratio'] != 0)
    
    if assignment_method == 'GMM':
        #fit a GMM to the rest of the values
        y = Y_df.loc[(has_idx & nz_idx), f'{col}_logratio'].values.reshape(-1,1)
        gm = GaussianMixture(n_components=2, random_state=274).fit(y)
        a, b = gm.means_.argsort(0).squeeze()
        #get assignments with high enough probability
        ps = gm.predict_proba(y)
        assignments = np.ones((len(ps)))
        assignments[ps[:,0] > GMM_tr] = a
        assignments[ps[:,1] > GMM_tr] = b
        assignments[(ps[:,1] < GMM_tr) & (ps[:,0] < GMM_tr)] = 0.5
    
    if assignment_method == 'manual':
        #use a manual logratio threshold
        low, high = assignment_tr
        y = Y_df.loc[(has_idx & nz_idx), f'{col}_logratio'].values
        assignments = np.ones((len(y))) * 0.5
        assignments[y < low] = 0
        assignments[y > high] = 1
        
    if assignment_method == 'sigma':
        #sequences a fraction of a stdev around 0 will be set to not confident
        assert type(assignment_tr) == float, 'only pass a single float if method is assignment method is sigma'
        y = Y_df.loc[(has_idx & nz_idx), f'{col}_logratio'].values
        sigma = y.std()
        assignments = np.ones((len(y))) * 0.5
        assignments[y > (sigma*assignment_tr)] = 1
        assignments[y < -(sigma*assignment_tr)] = 0
                              
    #add to the main df
    Y_df.loc[(has_idx & nz_idx), f'{col}_enriched'] = assignments
    Y_df.loc[Y_df[f'{col}_enriched'] == 0.5, f'{col}_enriched'] = 'not confident'
    
    #add a train test split
    if split_data:
        Y_df[f'{col}_split'] = np.nan
        all_idx = Y_df.index[(Y_df[f'{col}_enriched'] == 1.0) | (Y_df[f'{col}_enriched'] == 0.0)]
        train_idx, test_idx = train_test_split(all_idx,
                                               test_size=test_size, random_state=42)
        Y_df.loc[train_idx, f'{col}_split'] = 'train'
        Y_df.loc[test_idx, f'{col}_split'] = 'test'
        
    #plot to check
    if plot_check:
        #seaborn ignores data with NaN in the hue column
        sns.displot(data=Y_df, x=f'{col}_logratio', hue=f'{col}_enriched', palette=plot_kwargs['palette'])
        if 'save_dir' in plot_kwargs:
            save_fnm = os.path.join(plot_kwargs['save_dir'], f'{col}_enrichment_dist.pdf')
            plt.savefig(save_fnm)
    #how many of each class
    print (col)
    if only_ovlp:
        print('{} seqs shared in rounds'.format(sum(has_idx)))
    print('{} seqs total'.format(sum(~Y_df[f'{col}_split'].isna())))
    print('{} seqs not assigned (low confidence)'.format(sum(Y_df[f'{col}_enriched'] == 'not confident')))
    print('{} seqs not enriched'.format(sum(Y_df[f'{col}_enriched'] == 0)))
    print('{} seqs enriched'.format(sum(Y_df[f'{col}_enriched'] == 1)))
    print('')
    return (Y_df)



### FUNCTIONS FOR EXPLORING COUNTS DISTRIBUTIONS



def get_campaign_info_from_fasta_dict(label_dict):
    '''
    Inputs
    
        label_dict: a dictionary keyed by the label of the round, and the value being the fnm
        
    Output
    
        r_info: a df with a column for each round's counts
    '''
    r_info = []
    for seq_round, fname in label_dict.items():
        seq_info = {}
        #parse fasta file
        with open(fname) as fasta_file:
            for title, seq in SimpleFastaParser(fasta_file):
                count = int(title.split("-")[-1])
                if seq in seq_info.keys():  # no duplicate sequences allowed
                    raise Exception("At {}, found second instance of sequence {}".format(title, seq))
                seq_info[seq] = {"{}_ct".format(seq_round): count, 
                                 "{}_id".format(seq_round): title}
        seq_info = pd.DataFrame.from_dict(seq_info, orient='index')
        seq_info.index.name = 'seq'
        r_info.append(seq_info)
    r_info = reduce(lambda x, y: pd.merge(x, y, how='outer', left_index=True, right_index=True), r_info)
    return r_info


def process_round_info(r_info, round_labs, normalize=False, fill_count=1, new_norm=False, fill_abund=None):
    
    '''
    Process the counts from a df containing the rounds from a campaign
    
    Inputs:
        r_info: the df containing the count data (indexed by each sequence)
        round_labs: a list containing the round names (matching the dict names passed to 
                    the get_campaign_info_from_fasta_dict)
        normalize: normalize the counts to a fraction of the whole round library size
                    and multiple by a factor get it back to the range of sequencing counts
        fill_count: lowest count threshold. Below this values will be clipped
        new_norm: new method for normalizing the data in which we clip the counts
                    before normalizing by the library size (total counts)
        
    Outputs:
        r_info: new df with added columns
        
    Work to be done on this:
        make it so that you don't need to hard code the number of rounds
    
    '''
        
    print("total reads:")
    total_reads = r_info[ [f'{r}_ct' for r in round_labs] ].sum(axis=0)
    print(total_reads)
    print("average reads: {:0.1f}".format(total_reads.mean()))
    
    #record whether the round has the sequence
    for r in round_labs:
        r_info[f'{r}_hasread'] = r_info[f'{r}_ct'] >= fill_count
    
    print ('Number of unique reads:')
    print(r_info[[f'{r}_hasread' for r in round_labs]].sum(axis=0))
    
    #normalize the library size of each round to be the same
    if normalize:
        if type(normalize) != bool:
            norm_target = normalize
        else:
            norm_target = 1e6 #r_info[ [f'{r}_ct' for r in round_labs] ].sum(axis=0).mean()            
            
        if new_norm:
            for r in round_labs:
                #calculate the library size for normalizing
                lib_size = r_info[f'{r}_ct'].sum()
                
                #clip the floor based on a minimal abundance value
                if fill_abund:
                    #normalize by library size
                    r_info[f'{r}_normct'] = r_info[f'{r}_ct'] * norm_target / lib_size
                    #clip the abundance value
                    r_info[f'{r}_normct'] = r_info[f'{r}_normct'].clip(lower=fill_abund).fillna(fill_abund)
                
                #clip based on a minimal count
                else:
                    #clip counts
                    r_info[f'{r}_normct'] = r_info[f'{r}_ct'].clip(lower=fill_count).fillna(fill_count)
                    #normalize by library size
                    r_info[f'{r}_normct'] = r_info[f'{r}_normct'] * norm_target / lib_size
                    
                #log transform the counts
                r_info[f'{r}_logct'] = np.log10(r_info[f'{r}_normct'])
        else:
            for r in round_labs:
                r_info[f'{r}_normct'] = r_info[f'{r}_ct'] * norm_target / r_info[f'{r}_ct'].sum()
                r_info[f'{r}_normct'] = r_info[f'{r}_normct'].clip(lower=fill_count).fillna(fill_count)
                r_info[f'{r}_logct'] = np.log10(r_info[f'{r}_normct'])
    
    else:
        for r in round_labs:
            r_info[f'{r}_logct'] = np.log10(r_info[f'{r}_ct'].clip(lower=fill_count).fillna(fill_count))
            
    #calculate ratios
    pairs = list(combinations(round_labs, 2))
    for ra,rb in pairs:
        r_info[f'{rb}_{ra}_logratio'] = r_info[f'{rb}_logct'] - r_info[f'{ra}_logct']
        
    r_info = r_info.sort_values([f'{r}_ct' for r in round_labs], ascending=False)

    return r_info

def overlap_between_rounds(subdf):
    
    '''
    Get the overlap in number of unique sequences found in the rounds
    and report as a table
    
    Inputs:
        subdf: a df only containing boolean columns 
    '''

    o_df = np.zeros((subdf.shape[1], subdf.shape[1]))
    for ii,col1 in enumerate(subdf.columns):
        for jj, col2 in enumerate(subdf.columns):
            o_df[ii,jj] = (subdf[col1] & subdf[col2]).sum()

    o_df = pd.DataFrame(o_df, index=subdf.columns, columns=subdf.columns)
    
    print ('Num sequences in all rounds: {}'.format(subdf.apply(all, 1).sum()))
    
    return(o_df)    



def venn_diagram_vals(subdf):
    '''
    Function for getting the entries of a full venn diagram between rounds
    It corrects each entry of a VD by the value of the intersection
    
    Inputs: 
        subdf: a df only containing boolean columns 
    '''
    
    #retrieve the rounds in this df
    round_labs = list(subdf.columns.str.strip('_hasread'))

    #gets the overlap between columns
    def get_overlap(subdf, rnds):
        sdf = subdf[[f'{r}_hasread' for r in rnds]]
        return (sdf.apply(all, 1).sum())

    #get all possible combinations of rounds
    all_pairs = [list(combinations(round_labs, k+1)) for k in range(len(round_labs))]
    #reverse to start with the most intersection
    all_pairs = all_pairs[::-1]

    #get the uncorrected overlap values
    vd = {}
    #go through each order of combinations separately
    for il, layer in enumerate(all_pairs):
        vd[il] = {}
        for section in layer:
            sct_label = '_'.join(section)
            vd[il][sct_label] = get_overlap(subdf, section)

    #correct the overlap in layers
    cvd = vd.copy()
    for il in range(1,len(all_pairs)):
        #for all the sections in this layer remove values from the above layer
        for sct in all_pairs[il]:
            #go through the above layers and remove
            for up_l in range(il):
                #check section by section
                for up_sct in all_pairs[up_l]:
                    if set(sct).issubset(set(up_sct)):
                        #correct the counts
                        cvd[il]['_'.join(sct)] = cvd[il]['_'.join(sct)] - cvd[up_l]['_'.join(up_sct)]
                        
    return (cvd)
    
    

def plot_count_distribution(r_info, round_labs, count_metric='normct', setxrange=False,
                            plotkwargs={'kde':False, 'hist_kws':{'log':True}}):
    '''
    Plot a histogram for each round of the relevant count metric
    
    Input:
        setxrange: select the x range to be from 0 to the largest valueto be plotted
    
    '''
    if setxrange:
        if (type(setxrange) == float) | (type(setxrange) == int):
            xmax = setxrange
        elif type(setxrange) == bool:
            #get the highest value in the df
            subdf = r_info[[f'{r}_{count_metric}' for r in round_labs]]
            xmax = subdf.max().max()
    
    for r in round_labs:
        if (type(setxrange) == float) | (type(setxrange) == int):
            tmp = r_info[f'{r}_{count_metric}'][r_info[f'{r}_{count_metric}'] <= xmax]
        else:
            tmp = r_info[f'{r}_{count_metric}']
        sns.distplot(x=tmp, **plotkwargs).set(
            title=f'{r} {count_metric} distribution')
        if setxrange:
            plt.xlim([0, xmax])
        plt.show()
        
        
        
        
def hamming_distribution_from_consensus(r_info, consensus_metric=None, consensus_seq=None,
                                       plotkwargs = {'kde':False, 'hist_kws':{'log':True}}):
    
    '''
    Calculates the hamming distance distributon wrt to a consensus sequence
    and plots the distribution
    Requires use of aligned sequences
    
    Input:
        r_info: the df 
        consensus_metric: if no consensus passed, the column with which to select
                            the most abundant sequence
        consensus_seq: the consensus sequence to use
        
    Output:
        plots the distribution
        hamming_dist: a series containing the values of all the hamming dists
    
    '''

    print ("Number of total unique sequences:", r_info.shape[0])
    
    #if no selected consensus sequence select the most abundant
    if consensus_seq is None:
        consensus_seq = r_info[consensus_metric].idxmax()

    hamming_dist = r_info.index.map(lambda x: sum(ch1 != ch2 for ch1, ch2 in zip(x, consensus_seq)))
    
    fig, ax = plt.subplots(1,1, figsize=(4,3))
    ax = sns.distplot(hamming_dist, ax=ax, **plotkwargs)
    ax.set_xlabel(f'Hamming distance from most abundandt sequence in {consensus_metric}')
    ax.set_ylabel('number of unique sequences')
    plt.show()
    
    return (hamming_dist)



def plot_length_dists(r_info, round_labs,
                      plotkwargs = {'kde':False}):
    '''
    Plots the distribution of sequence lengths for each round
    '''
    
    for r in round_labs:
        seqs = pd.Series(r_info[f'{r}_hasread'].index)
        lens = seqs.apply(len).values
        sns.distplot(a =lens, **plotkwargs).set(
            xlabel = 'Length', title=r)
        plt.show()
        

def plot_enrichment_distributions(r_info, round_labs, ovlp=False, no_zero=True,
                                  plotkwargs={'kde':True}):
    
    '''
    Plots the histograms for the enrichment ratios
    Plots all pairs of enrichment ratios for rounds in round_labs
    
    '''

    plot_pairs = combinations(round_labs, 2)
    for pair in plot_pairs:
        a,b = pair
        if ovlp:
            r_info = r_info[r_info[f'{a}_hasread'] & r_info[f'{a}_hasread']]
        if no_zero:
            r_info = r_info[r_info[f'{b}_{a}_logratio'] != 0]
        x = r_info[f'{b}_{a}_logratio']
        sns.distplot(a=x,**plotkwargs)
        plt.show()
        
        
        
#### FUNCTIONS ON SEQUENCES

def CDR_annot(imgt):
    cdr_range=[('27', '38'), ('56', '65'), ('105', '117')]
    imgtsrch = re.search('\d+', imgt)
    if not imgtsrch:
        return (0)
    imgtnum = int(imgtsrch.group(0))
    for i,(l,r) in enumerate(cdr_range):
        if (imgtnum >= int(l)) & (imgtnum <= int(r)):
            return (int(i+1))

def import_wt_from_align(align_fnm: str):
    
    wa_df = extract_cdrs(align_fnm)
    #extract the IMGT indices for the sequence
    w_ser = wa_df.iloc[:, wa_df.columns.str.match('\d+')].iloc[0]
    wt_seq = ''.join(w_ser[w_ser != '-'].values)
    imgt_col = w_ser.index[w_ser != '-'].values
    
    return (wt_seq, imgt_col)


def make_singles_muts(align_fnm: str, wt_name: str, with_WT: bool) -> tuple:
    '''
    Generates a list of single point mutants from a wild-type (WT) sequence.
    The sequences are labelled using the sequence indices and also IMGT indices
    sequence labels in the fasta file are of the form {WT_name}_{wt_AA}{wt_idx}{mut_AA}_({IMGT_idx})

    Args:
        align_fnm (str): The file name of the alignment file containing the WT sequence.
        wt_name (str): A label for the WT sequence.
        with_WT (bool): Whether the WT sequence is retained in the file

    Returns:
        A tuple containing three elements:
            - A pandas Series object containing the mutant sequences and their labels.
            - The WT sequence as a string.
            - A numpy array containing the indices of the IMGT columns in the alignment file.
    '''
    
    wt_seq, imgt_col = import_wt_from_align(align_fnm)

    #make mutants and label them accordingly
    alphabet, _ = import_AA_alphabet(withgaps=False)

    mut_seqs, mut_labels = [], []
    if with_WT:
        mut_seqs.append(wt_seq)
        mut_labels.append(f'{wt_name}_{wt_seq[0]}{1}{wt_seq[0]}_({imgt_col[0]})')
    for ii, s in enumerate(wt_seq):
        mut_alph = list(alphabet)
        mut_alph.remove(s)
        for jj, ms in enumerate(mut_alph):
            #make a copy of the sequence
            mut_seq_list = list(wt_seq)
            mut_seq_list[ii] = ms
            mut_seq = ''.join(mut_seq_list)
            mut_seqs.append(mut_seq)
            #create the sequence label
            mut_lab = f'{wt_name}_{s}{ii+1}{ms}_({imgt_col[ii]})'
            mut_labels.append(mut_lab)
            
    muts = pd.Series(mut_seqs, mut_labels)
    
    return (muts, wt_seq, imgt_col)



def make_double_muts(align_fnm: str, wt_name: str, with_WT: bool) -> tuple:
    '''
    Generates a list of double point mutants from a wild-type (WT) sequence.
    The sequences are labelled using the sequence indices and also IMGT indices
    sequence labels in the fasta file are of the form {WT_name}_{wt_AA1}{wt_idx1}{mut_AA1}_({IMGT_idx1})_{wt_AA2}{wt_idx2}{mut_AA2}_({IMGT_idx2})

    Args:
        align_fnm (str): The file name of the alignment file containing the WT sequence.
        wt_name (str): A label for the WT sequence.
        with_WT (bool): Whether the WT sequence is retained in the file

    Returns:
        A tuple containing three elements:
            - A pandas Series object containing the mutant sequences and their labels.
            - The WT sequence as a string.
            - A numpy array containing the indices of the IMGT columns in the alignment file.
    '''
    
    wt_seq, imgt_col = import_wt_from_align(align_fnm)

    #make mutants and label them accordingly
    alphabet, _ = import_AA_alphabet(withgaps=False)

    mut_seqs, mut_labels = [], []
    if with_WT:
        mut_seqs.append(wt_seq)
        mut_labels.append(f'{wt_name}_{wt_seq[0]}{1}{wt_seq[0]}_({imgt_col[0]})_{wt_seq[0]}{1}{wt_seq[0]}_({imgt_col[0]})')
    
    counter, ST = 0, time.time()
    for i1, s1 in enumerate(wt_seq):
        for i2, s2 in enumerate(wt_seq):
            if i1 != i2:
                for j1, ms1 in enumerate(list(alphabet)):
                    for j2, ms2 in enumerate(list(alphabet)):
                        if (s1 != ms1) & (s2 != ms2):
                            #make a copy of the sequence
                            mut_seq_list = list(wt_seq)
                            #induce mutations
                            mut_seq_list[i1] = ms1
                            mut_seq_list[i2] = ms2
                            mut_seq = ''.join(mut_seq_list)
                            mut_seqs.append(mut_seq)
                            #create sequence label
                            mut_lab = f'{wt_name}_{s1}{i1+1}{ms1}_({imgt_col[i1]})_{s2}{i2+1}{ms2}_({imgt_col[i2]})'
                            mut_labels.append(mut_lab)
                            counter += 1

    print (f'Made {counter} double mutants, time elapsed: {time.time()-ST:.1f}', flush=True)
            
    muts = pd.Series(mut_seqs, mut_labels)
    
    return (muts, wt_seq, imgt_col)





from typing import List, Tuple
def induce_muts(muts: pd.DataFrame, align_fnm: str, wt_name: str) -> Tuple[pd.Series, str, List[int]]:
    """Create mutant sequences based on a given wildtype sequence and mutation data.

    Args:
        muts: A Pandas DataFrame containing mutation data, including the indices of the mutations.
        align_fnm: A string representing the filename of the sequence alignment file.
        wt_name: A string representing the name of the wildtype sequence.

    Returns:
        A tuple containing the following:
        - A Pandas Series containing the mutant sequences, with labels indicating the specific mutations made.
        - A string representing the wildtype sequence.
        - A list of integers representing the IMGT indices for the wildtype sequence.
    """
    
    wa_df = extract_cdrs(align_fnm)
    #extract the IMGT indices for the sequence
    w_ser = wa_df.iloc[:, wa_df.columns.str.match('\d+')].iloc[0]
    wt_seq = ''.join(w_ser[w_ser != '-'].values)
    imgt_col = list(w_ser.index[w_ser != '-'].values)

    #make mutants and label them accordingly
    alphabet, _ = import_AA_alphabet(withgaps=False)

    #check to see which indexing form to use
    if 'seq_ii' in muts.columns:
        use_ii = True
    if 'IMGT' in muts.columns:
        use_ii = False
    else:
        raise ('Need either seq_ii or IMGT in the mutations df as indices for the WT sequence')

    mut_seqs, mut_labels = [], []
    for ii, row in muts.iterrows():
        mut_seq_list = list(wt_seq)
        if use_ii:
            #note seq_ii is 1-indexed
            wt_aa = mut_seq_list[row['seq_ii'] -1]
            mut_seq_list[row['seq_ii'] -1] = row['AA']
        else:
            wt_aa = mut_seq_list[imgt_col.index(row['IMGT'])]
            mut_seq_list[imgt_col.index(row['IMGT'])] = row['AA']
        mut_seq = ''.join(mut_seq_list)
        mut_seqs.append(mut_seq)
        #create the sequence label
        mut_lab = f"{wt_name}_{wt_aa}{imgt_col.index(row['IMGT']) + 1}{row['AA']}_({row['IMGT']})"
        mut_labels.append(mut_lab)

    mut_series = pd.Series(mut_seqs, mut_labels)
    
    return (mut_series, wt_seq, imgt_col)


def id_mut_from_seq(wt:str, mut:str, imgt=None, justnames=True):
    '''
    Helper function to get the mutations that differentiate a mutant
    from a wt sequence
    
    Input:
        wt, mut: strings containing the sequences. Must be the same length
    
    Optional:
        imgt (array): the imgt numbers in order for each WT residue
        justnames: just return a list of the mutation names
    
    Returns:
        pd.DataFrame: cols: [idx, wtAA, mutAA, mutName] + [imgt] (optional)
    '''
    if len(wt) != len(mut):
        raise ValueError('sequences must be the same length')
        
    res_list = []
    for i,(w,m) in enumerate(zip(wt, mut)):
        if w != m:
            if imgt is not None:
                res_list.append([i+1,w,m,imgt[i]])
            else:
                res_list.append([i+1,w,m])
    if imgt is not None:
        res_df = pd.DataFrame(res_list, columns=['idx', 'wtAA', 'mutAA', 'imgt'])
    else:
        res_df = pd.DataFrame(res_list, columns=['idx', 'wtAA', 'mutAA'])
        
    res_df['mutname'] = res_df['wtAA'] + res_df['idx'].astype(str) + res_df['mutAA']
    
    if justnames:
        return (res_df['mutname'].values)
    else:
        return (res_df)
    
    
    
def unpack_mutname(df, s_col=None, fmt='NAME_WIM_(I)'):
    '''
    extract the metadata from a mutation name
    
    Inputs:
        df: the df containg the mutnames
        s_col: which column contains the mutnames
        fmt: which mutname format to use
    
    supported mutation name formats:
        NAME_WIM_(I): {NAME}_{WT_AA}{SEQIDX}{MUT_AA}_({IMGT})
        NAME_WIM_(I)_WIM_(I): same as above but for double mutants
    '''
    if not isinstance(df, pd.DataFrame):
        try:
            df = pd.DataFrame(df, columns=['mutname'])
        except ValueError:
            raise ValueError("Please pass a df containing the mutnames or some array-like object")
            
    s = df[s_col]
    if fmt == 'NAME_WIM_(I)':
        name = s.str.extract('(\w+)_\w\d+\w_\(\w+\)')
        wt_aa = s.str.extract('\w+_(\w)\d+\w_\(\w+\)')
        seqidx = s.str.extract('\w+_\w(\d+)\w_\(\w+\)')
        mut_aa = s.str.extract('\w+_\w\d+(\w)_\(\w+\)')
        imgt = s.str.extract('\w+_\w\d+\w_\((\w+)\)')
        
        df['seq_name'] = name
        df['wt_AA'] = wt_aa
        df['seqidx'] = seqidx
        df['mut_AA'] = mut_aa
        df['IMGT'] = imgt
        
    elif fmt == 'NAME_WIM_(I)_WIM_(I)':
        name = s.str.extract('(\w+)_\w\d+\w_\(\w+\)_\w\d+\w_\(\w+\)')
        
        wt_aa1 = s.str.extract('\w+_(\w)\d+\w_\(\w+\)_\w\d+\w_\(\w+\)')
        seqidx1 = s.str.extract('\w+_\w(\d+)\w_\(\w+\)_\w\d+\w_\(\w+\)')
        mut_aa1 = s.str.extract('\w+_\w\d+(\w)_\(\w+\)_\w\d+\w_\(\w+\)')
        imgt1 = s.str.extract('\w+_\w\d+\w_\((\w+)\)_\w\d+\w_\(\w+\)')
        
        wt_aa2 = s.str.extract('\w+_\w\d+\w_\(\w+\)_(\w)\d+\w_\(\w+\)')
        seqidx2 = s.str.extract('\w+_\w\d+\w_\(\w+\)_\w(\d+)\w_\(\w+\)')
        mut_aa2 = s.str.extract('\w+_\w\d+\w_\(\w+\)_\w\d+(\w)_\(\w+\)')
        imgt2 = s.str.extract('\w+_\w\d+\w_\(\w+\)_\w\d+\w_\((\w+)\)')
        
        df['seq_name'] = name
        df['wt_AA1'] = wt_aa1
        df['seqidx1'] = seqidx1
        df['mut_AA1'] = mut_aa1
        df['IMGT1'] = imgt1

        df['wt_AA2'] = wt_aa2
        df['seqidx2'] = seqidx2
        df['mut_AA2'] = mut_aa2
        df['IMGT2'] = imgt2
    
    return (df)


def _wrap_str(s, wrap_width=100):
    wrapped_s = []
    i = 0
    while i < len(s):
        wrapped_s.append(s[i:(i+wrap_width)])
        i += wrap_width
    return ([''.join(ii) for ii in wrapped_s])

def format_seq(seq, indices,
                interval=10, wrap_width=70, start_i=4):
    
    wt_seq, imgt_col = seq, indices
    seqlen = len(wt_seq)
    ii_str = np.array(list(' '*seqlen))
    #add in the middle indices
    i = start_i
    while (i+interval) < seqlen:
        index = imgt_col[i]
        index_str = list(index)
        index_len = len(index_str)
        ii_str[i: (i+index_len)] = index_str
        i += interval

    #add the first index
    i0 = 0
    index_0_str = list(imgt_col[i0])
    ii_str[i0: (i0+len(index_0_str))]  = index_0_str

    #add the last index
    il = len(imgt_col)
    index_l_str = list(imgt_col[-1])
    ii_str[-(len(index_l_str)):]  = index_l_str

    #wrap at a particular width
    wrap_ii_line = _wrap_str(ii_str, wrap_width)
    wrap_seq_line = _wrap_str(list(wt_seq), wrap_width)

    for l in range(len(wrap_ii_line)):
        print (wrap_ii_line[l])
        print (wrap_seq_line[l])