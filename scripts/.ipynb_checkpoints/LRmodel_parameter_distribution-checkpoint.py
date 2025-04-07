'''
Script for looking at where particular mutations sit
in the distribution of weights from a trained LR model
Inputs:

    model_pth: the path to the model whose paraemeters we'll use
    fig_svpth: the path where the figures will be saved
    mut_fnm: the path to the csv containg the mutations to test
                The csv should contain the following columns:
                Mut: the AA of the mutation
                IMGT: the IMGT idx for the site
                idx: if not the IMGT the index in the context of the WT
    only_sites: if true signals that the mutation csv only contains sites
                and to plot all the weights for that site. The csv should
                then contain
                IMGT: the IMGT idx for the sites we want to plt
                idx: if not the IMGT the index in the context of the WT
    conv_imgt: if used, signals that the mutations are in the sequence
                indexing and need to be converted to IMGT indexing
    wt_fnm: the filename of the WT sequence alignment to use for original
                indexing
    model_nm: the name of the model to use in figure titles
'''

import os, sys, re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from argparse import ArgumentParser

sys.path.append('/n/groups/marks/projects/binding_affinity')
import affinity_maturation_utilities as util
from modelling_functions import *


parser = ArgumentParser()
parser.add_argument("--model_pth", type=str)
parser.add_argument("--model_scores_pth", type=str)
parser.add_argument("--fig_svpth", type=str)
parser.add_argument("--mut_fnm", type=str)
parser.add_argument("--only_sites", action='store_true', default=False)
parser.add_argument("--conv_imgt", action='store_true', default=False)
parser.add_argument("--wt_fnm", type=str)
parser.add_argument("--model_nm", type=str)
args = parser.parse_args()


#arrange savepaths
if not args.fig_svpth:
    args.fig_svpth = os.path.splitext(args.model_pth)[0] + '.png'

#import model and collect parameters
if args.model_pth:
    load_model = joblib.load(args.model_pth)
    coef0 = load_model.model.coef_
    try:
        c_df = load_model.imgt_aa_col
    except AttributeError:
        print ("trainer object needs to have a stored imgt_aa_col df detailing which parameter corresponds to which residue in the sequence")
        raise
    c_df['beta'] = np.squeeze(coef0)
if args.model_scores_pth:
    c_df = pd.read_csv(args.model_scores_pth)

#load the mutations to test
if args.mut_fnm:
    mut_df = pd.read_csv(args.mut_fnm)
else:
    mut_df = pd.DataFrame([], columns=['idx', 'Mut'])

if args.conv_imgt:

    #import the WT seq
    wt_df = pd.read_csv(args.wt_fnm)
    assert (wt_df.shape[0]) == 1
    wt_s = wt_df.iloc[:, wt_df.columns.str.match('\d+')]

    #construct a key to get back to the WT seq idx
    wt_has_cols = list(wt_s.columns[~(wt_s.iloc[0] == '-')])
    IMGT_WT_idx_conv = {v:(i+1) for i,v in enumerate(wt_has_cols)}
    WT_idx_IMGT_conv = pd.DataFrame(wt_has_cols, columns=['IMGT'])
    WT_idx_IMGT_conv.index = WT_idx_IMGT_conv.index + 1
    
    mut_df['IMGT'] = [WT_idx_IMGT_conv.loc[i, 'IMGT'] for i in mut_df['idx']]
    c_df['seq_ii'] = [IMGT_WT_idx_conv[i] for i in c_df['IMGT']]


#select the mutations to plot
if not args.only_sites:
    mut_df['IMGT_AA'] = mut_df['IMGT'] + '_' + mut_df['Mut']
    c_df['inMut'] = c_df['IMGT_AA'].isin(mut_df['IMGT_AA'])
else:
    c_df['inMut'] = c_df['IMGT'].isin(mut_df['IMGT'])

#check how many mutations are actually in the model
print ('The following mutations are in the model')
print(list(c_df.loc[c_df['inMut'], 'IMGT_AA']))

#save the parameters
save_df = c_df[c_df['beta'] != 0].sort_values('beta', ascending=False).reset_index(drop=True)

#print some summary statistics
print ('Number of non zero parameters:')
print (save_df.shape[0])
print ('Percentile of each test mutation:')
tmp = save_df[save_df['inMut']].copy()
tmp['perc'] = ((1-  tmp.index.values / save_df.shape[0])*100).round(2)
tmp_d = {f"{r['seq_ii']}{r['AA']}":r['perc'] for ii, r in tmp.iterrows()}
print (tmp_d)
top = 10
print ('Number of hits recalled in the top {} mutations'.format(top))
recall = ((1-tmp['perc']/100)*save_df.shape[0] < top).sum()
print (recall)

if args.model_pth:
    save_pth = args.model_pth.strip('.sav') + '_params.csv'
    save_df.to_csv(save_pth, index=False)

fig = plt.figure()
sns.set(font_scale=1)
sns.set_style('ticks')
sns.histplot(save_df['beta'], log_scale=(False, False), binwidth=0.25)
for b in c_df.loc[c_df['inMut'], 'beta']:
    plt.axvline(b, linestyle='dotted', c='red')
plt.title(f'Parameter distribution for \n {args.model_nm}')
plt.xlabel("Weight")
plt.savefig(fname=args.fig_svpth, pad_inches=0, dpi=fig.dpi)