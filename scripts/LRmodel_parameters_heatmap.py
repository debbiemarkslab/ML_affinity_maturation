import os, sys, re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from argparse import ArgumentParser

sys.path.append('scripts')
import affinity_maturation_utilities as util
from modelling_functions import *


def main():
    parser = ArgumentParser()
    parser.add_argument("--model_pth", type=str, help="the path to the model whose parameters we are plotting")
    parser.add_argument("--wt_alignment", type=str, help='path to aligned wt sequence for converting to imgt numbering')
    parser.add_argument("--fig_svpth", type=str, default=None, help="the path where the figures will be saved")
    parser.add_argument("--model_nm", help='the name of the model for figures')
    parser.add_argument("--train_subseq", type=str, help="which subsequence of the antibody the model was trained on", default='FullSeqs_nogaps')
    parser.add_argument("--nogaps", action='store_true', default=False, help='whether the model alphabet contained gaps', default=True)
    args = parser.parse_args()

    #arrange savepaths
    if not args.fig_svpth:
        args.fig_svpth = os.path.splitext(args.model_pth)[0] + '.png'


    #import model and collect parameters
    load_model = joblib.load(args.model_pth)
    coef0 = load_model.model.coef_
    try:
        c_df = load_model.imgt_aa_col
    except AttributeError:
        print ("trainer object needs to have a stored imgt_aa_col df detailing which parameter corresponds to which residue in the sequence")
        raise
    c_df['beta'] = np.squeeze(coef0)
    
    #import the WT alignment
    wt_afnm = os.path.join(args.exp_dir, 'aligned_WT_sequence.csv')
    wt_adf = pd.read_csv(wt_afnm)
    imgt_col = wt_adf.columns[wt_adf.columns.str.contains('\d+')].tolist()
    imgt_col = pd.Series(imgt_col)[(wt_adf.loc[:,imgt_col]).values[0] != '-'].reset_index(drop=True)
    imgt_ii_dict = {IMGT:II for II,IMGT in enumerate(imgt_col)}

    alphabet, D = import_AA_alphabet(withgaps=(not args.nogaps))

    #arrange parameters
    c_dfc = pd.pivot_table(c_df, index='IMGT', columns='AA', values='beta')
    try:
        c_dfc = c_dfc.loc[imgt_col, :]
    except KeyError:
        print ("IMGT col from the subsequence key not present in trained parameter key")
        print ("The following IMGT columns are not present in the model parameters")
        print (list(c_dfc.columns[~pd.Series(c_dfc.columns).isin(imgt_col)]))
        raise
    try:
        c_dfc = c_dfc.loc[:, list(alphabet)]
    except KeyError:
        print ("AA letter in alphabet not present in trained parameter key")
        print ("The following residues are not present in the model parameters")
        print (list(c_dfc.index[~pd.Series(c_dfc.index).isin(list(alphabet))]))
        raise   
    c_dfc = c_dfc.T


    #Plot heatmap  
    
    fig = plt.figure(figsize=(17, 6))
    sns.set_theme(style='white', font_scale=1.4)
    g = sns.heatmap(c_dfc, cmap='RdBu', center=center)
    g.set(ylabel='Mutant AA')
    g.set_facecolor('grey')
    g.set_xlabel('')
    plt.title(f'Model parameters for \n {args.model_nm}')
    plt.savefig(fname=args.fig_svpth, pad_inches=0, dpi=fig.dpi)
    
    #add in patches to demarcate CDRs
    cdr_range=[('27', '38'), ('56', '65'), ('105', '117')]
    for l,r in cdr_range:
        lii, rii = imgt_ii_dict[l], imgt_ii_dict[r]
        space = rii-lii
        g.add_patch(Rectangle((lii,0), space, len(alphabet)-0.1,
                              fill=False, edgecolor='black', lw=1,
                             linestyle='solid'))

if __name__ == '__main__':
    main()