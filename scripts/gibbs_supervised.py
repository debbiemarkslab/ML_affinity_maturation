import os, sys, re
import affinity_maturation_utilities as util
from modelling_functions import *
from argparse import ArgumentParser
import joblib
import seaborn as sns
import torch

sys.path.append('ML_affinity_maturation/scripts')
from spec import *

from copy import deepcopy
import time as time

from dataset_classes import OneHotArrayDataset
from supervised_model_classes import CNN, MLP, LSTM, get_latest_ckpt

from gibbs_utils import *

T_cycle_options = [(5,1.0,0.5),
                   (1,0.5,0.2),
                   (0.5,0.2,0.1)]

def main():
    
    run_parser = ArgumentParser()
    run_parser.add_argument('--proj',
                           default='RX002')
    run_parser.add_argument('--svnm',
                           default='./tmp.csv')
    run_parser.add_argument('--wt_fnm', type=str)
    run_parser.add_argument('--ckptfnm')
    run_parser.add_argument('--numsteps', type=int,
                            default=100)
    run_parser.add_argument('--numseqs',
                           default=100, type=int)
    run_parser.add_argument('--maxham',
                            default = 7, type=int)
    run_parser.add_argument('--minham',
                            default = 0, type=int)
    run_parser.add_argument('--hamweight',
                            default = 1000, type=int)
    run_parser.add_argument('--modeltype',
                            default='LogisticRegression')
    run_parser.add_argument('--temp_set',
                            default=1, type=int)
    run_parser.add_argument('--seed',
                            default=42, type=int)
    run_args = run_parser.parse_args()
    
    #get arguments
    parser = get_training_parser()
    args = parser.parse_args('')
    args = update_args(run_args.proj, args)
    if not os.path.isdir(os.path.dirname(run_args.svnm)):
        os.makedirs(os.path.dirname(run_args.svnm))

    #import data
    data = data_importer(args)

    #import wt
    
    wt_seq, imgt_cols = util.import_wt_from_align(run_args.wt_fnm)
    imgt_ser = pd.Series(imgt_cols).reset_index().set_index(0)
    imgt_ser['site'] = imgt_ser['index'] + 1
    alphabet, _ = import_AA_alphabet(withgaps=False)

    #get the indices of framework residues
    cdr_range=[('27', '38'), ('56', '65'), ('105', '117')]
    cdr_i_sites = [list_from_range([(int(imgt_ser.loc[l,'site']),
                                     int(imgt_ser.loc[r,'site']))])
                                        for l,r in cdr_range]
    fr_range=[('1','26'),('39','55'),('66','104'),('118',imgt_cols[-1])]
    fr_sites = list_from_range([(int(imgt_ser.loc[l,'site']),
                                     int(imgt_ser.loc[r,'site']))
                                        for l,r in fr_range])
    
    #select the amino acids to restrict
    restricted_aa = ['C', 'M']

    ##### RUN SAMPLING #######

    #init
    
    if isinstance(run_args.numsteps, int):
        m = run_args.numsteps
    T_cycle = []
    for t in T_cycle_options[run_args.temp_set]:
        T_cycle.extend([t]*m)
    numseqs = run_args.numseqs

    logiter = 10

    if run_args.modeltype == 'LogisticRegression':
        load_trainer = joblib.load(run_args.ckptfnm)
        scorer = Logistic_Regression_Scorer(load_trainer.model, data, args)
    elif run_args.modeltype == 'CNN':
        model_fnm = os.path.join(run_args.ckptfnm, get_latest_ckpt(run_args.ckptfnm))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        load_model = CNN(len(wt_seq))
        load_model.load_state_dict(torch.load(model_fnm))
        scorer = CNN_Scorer(load_model, device)
    else:
        raise (ValueError('Pass a known modeltype to the arguments'))
    sampler = Sampler(scorer, args, alphabet, wt_seq)
    fixed_sites = deepcopy(fr_sites)

    np.random.seed(run_args.seed)
    sample_res = []
    sum_cols = ['startSeq',
            'sampleSeq', 'hamming', 'score',
                'H1muts', 'H2muts', 'H3muts', 'mutnames',
           'bestSeq', 'besthamming', 'bestscore',
                'bestH1muts', 'bestH2muts', 'bestH3muts', 'bestmutnames']
    #sample a sequence
    tST = time.time()
    for nii in range(numseqs):
        ST = time.time()
        #run cycle
        report = []
        report_cols = ['seq0', 'seq1', 'ii', 'start_aa', 'new_aa', 'wt_aa', 'start_score', 'new_score']

        all_scores = []
        seq0 = deepcopy(wt_seq)
        for n,T in enumerate(T_cycle):
            hargs = dict(min_hamming=run_args.minham, max_hamming=run_args.maxham, weight=run_args.hamweight)
            res = sampler.step(seq0, T=T,
                               fixed_sites = fixed_sites,
                               restricted_aa = restricted_aa,
                               hamming_penalty=True, **hargs)
            new_seq, seq_ii, seq_aa, start_aa, wt_aa, start_score, new_score, scores = res
            report.append([seq0, new_seq, seq_ii, start_aa, seq_aa, wt_aa, start_score, new_score])
            seq0 = new_seq

            all_scores.append(scores)

        report = pd.DataFrame(report, columns=report_cols)
        sum_row = summarize_report(report, cdr_i_sites)
        sample_res.append(sum_row)
        RT = time.time()-ST
        tRT = time.time()-tST
        print ('Sample {}. Took {:.0f}s. Total time: {:.0f}s'.format(nii, RT, tRT), flush=True)
        if ((nii%logiter)==0) | ((nii+1) == numseqs):
            #save
            pd.DataFrame(sample_res, columns=sum_cols).to_csv(
                run_args.svnm, index=False
            )

if __name__ == '__main__':
    main()