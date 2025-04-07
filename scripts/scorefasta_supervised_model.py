#!/bin/python

import os, sys, re
sys.path.append('/n/groups/marks/projects/binding_affinity')
import affinity_maturation_utilities as util
from modelling_functions import *
from argparse import ArgumentParser
import joblib
import torch
import numpy as np
import pandas as pd

sys.path.append('/n/groups/marks/projects/binding_affinity/model_zoo')
from dataset_classes import OneHotArrayDataset
from supervised_model_classes import CNN, MLP, LSTM

def get_latest_ckpt(directory):
    ckpt_nms = os.listdir(directory)
    ckpt_nms = [c for c in ckpt_nms if re.search('.tar', c)]
    ckpt_epochs = pd.Series(ckpt_nms).str.extract('\w+_epoch(\d+)').iloc[:,0].values.astype(int)
    best_epoch_idx = np.argmax(ckpt_epochs)
    return (ckpt_nms[best_epoch_idx])

def score_df(df, load_model, device):
    dataset = OneHotArrayDataset(df,'seq',label=None)
    loader = torch.utils.data.DataLoader(dataset = dataset, batch_size = 128, shuffle = False)
    scores = []
    for seq_array, labels in loader:
        inputs = torch.reshape(seq_array,(seq_array.shape[0],seq_array.shape[2],seq_array.shape[1])).float().to(device)
        outputs = load_model(inputs).to(device)
        outputs = outputs.squeeze().tolist()
        scores.extend(outputs)

    return (scores)


def main():
    
    parser = ArgumentParser()
    parser.add_argument('--infasta', type=str,
                       help = 'the fnm of the fasta containing sequences to score')
    parser.add_argument('--ckpt_dir', type=str,
                       help='the directory containing all ckpts of your model. The latest one will be selected')
    parser.add_argument('--modeltype', type=str, default='CNN',
                        help = 'MLP or CNN architecture')
    parser.add_argument('--outfnm', type=str,
                        help='the filename for the csv output containing the scores')
    args = parser.parse_args()
    
    print (f'Scoring seqs from {args.infasta}')
    print (f'With {args.modeltype} model loaded from {args.ckpt_dir}')
    
    #import the sequences
    ST = time.time()
    test_res = util.import_fasta(args.infasta).reset_index().rename({'index':'mutname'}, axis=1)
    print ('Opening fasta: {:.1f}s'.format(time.time()-ST))
    
    model_fnm = os.path.join(args.ckpt_dir, get_latest_ckpt(args.ckpt_dir))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #load model architecture
    seqlen = len(test_res.iloc[0]['seq'])
    if args.modeltype == 'CNN':
        load_model = CNN(seqlen)
    elif args.modeltype == 'MLP':
        load_model = MLP(seqlen*20)

    #load parameters
    load_model.load_state_dict(torch.load(model_fnm))

    #score
    print ('Scoring...')
    ii, chunk_size, N = 0, 1024, test_res.shape[0]
    num_chunks = N//chunk_size
    all_scores = []
    counter, logcounter, ST = 0, 100, time.time()
    while ii < N:
        tmp = test_res.iloc[ii:(ii+chunk_size)]
        scores = score_df(tmp, load_model, device)
        all_scores.extend(scores)

        if (counter % logcounter) == 0:
            print ('Chunk {}/{} complete. Time elapsed: {:.1f}s'.format(counter, num_chunks, time.time()-ST), flush=True)

        counter += 1
        ii += chunk_size
        
    #add in
    test_res['model_score'] = all_scores
        
    #extract any metadata
    ST = time.time()
    test_res = util.unpack_mutname(test_res, 'mutname', fmt='NAME_WIM_(I)_WIM_(I)')
    print ('Unpacking metadata fasta: {:.1f}s'.format(time.time()-ST))
    
    #save
    print ('Saving...')
    outdir = os.path.dirname(args.outfnm)
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    test_res.to_csv(args.outfnm, index=False)
    
    
    
if __name__ == '__main__':
    main()