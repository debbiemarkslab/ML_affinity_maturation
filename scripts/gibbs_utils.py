import os, sys, re
sys.path.append('ML_affinity_maturation/scripts')
import affinity_maturation_utilities as util
from modelling_functions import *
from argparse import ArgumentParser
import joblib
import seaborn as sns
import torch


from spec import *

from copy import deepcopy
import time as time

from dataset_classes import OneHotArrayDataset
from supervised_model_classes import CNN, MLP, LSTM

def get_latest_ckpt(directory):
    ckpt_nms = os.listdir(directory)
    ckpt_nms = [c for c in ckpt_nms if re.search('.tar', c)]
    ckpt_epochs = pd.Series(ckpt_nms).str.extract('\w+_epoch(\d+)').iloc[:,0].values.astype(int)
    best_epoch_idx = np.argmax(ckpt_epochs)
    return (ckpt_nms[best_epoch_idx])

def list_from_range(arr: list):
    #arr: list of tuples with the left
    #and right index inclusive
    ii = []
    for l,r in arr:
        assert isinstance(l, int)
        assert isinstance(r, int)
        ii.extend([i for i in range(l, r+1)])
    return (ii)


def hamming(seq1,seq2):
    assert len(seq1) == len(seq2)
    h = 0
    for aa1,aa2 in zip(seq1,seq2):
        if aa1!=aa2:
            h+=1
    return h

class Sampler():
    '''optimize a focus sequence (uppercase only) according to statistical enery'''
    def __init__(self, scorer, args, alphabet, wt_seq):
        self.scorer = scorer
        self.args = args
        self.alphabet = alphabet
        self.wt_seq = wt_seq
        self.fixed_idx = None
        self.fixed_aa_idxs = None
        
        wt_idxs = []
        k = 0
        for aa in self.wt_seq:
            for ma in self.alphabet:
                if aa == ma:
                    wt_idxs.append(k)
                k+=1
        self.wt_idxs = np.array(wt_idxs)
        
    def get_fixed_idxs(self, fixed_sites):
        #fixed sites are sequence site numbers
        #starting from 1 rather than 0
        fixed_idxs = []
        k = 0
        for ii,aa in enumerate(self.wt_seq):
            for ma in self.alphabet:
                if (ii+1) in fixed_sites:
                    fixed_idxs.append(k)
                k+=1
        self.fixed_idx = fixed_idxs
        
    def get_restricted_aa(self, restricted_aa):
        #these AAs in the onehot will have
        #energies set to zero
        fixed_aa_idxs = []
        k = 0
        for ii,aa in enumerate(self.wt_seq):
            for ma in self.alphabet:
                if ma in restricted_aa:
                    fixed_aa_idxs.append(k)
                k+=1
        self.fixed_aa_idxs = fixed_aa_idxs
                
    
    def first_order_mutagenesis(self, seq):
        new_seqs = []
        new_names = []
        for i,aa in enumerate(seq):
            for ma in self.alphabet:
                new_seq = list(deepcopy(seq))
                new_seq[i] = ma
                new_seqs.append(''.join(new_seq))
                name = str(i+1) + ma
                new_names.append(name)

        return (new_seqs, new_names)
    
    def hamming(self, seq1,seq2):
        assert len(seq1) == len(seq2)
        h = 0
        for aa1,aa2 in zip(seq1,seq2):
            if aa1!=aa2:
                h+=1
        return h
    
    def all_hammings(self, seqs):
        h = np.array(list(map(lambda s: self.hamming(self.wt_seq, s), seqs)))
        return (h)
    
    def first_order_hamming_penalty(self, seqs, min_hamming, max_hamming, weight=10):
        ham_vec = self.all_hammings(seqs)
        penalty = -weight*(ham_vec<=min_hamming) - weight*(ham_vec>=max_hamming)
        
        return (penalty)
        
    def step(self,start_seq,T,fixed_sites=None, restricted_aa=None,
             hamming_penalty=False, **kwargs):
        '''introduce a mutation with boltzmann probability
        higher temperature = closer to uniform distribution'''
        
        if not 'min_hamming' in kwargs:
            kwargs['min_hamming'] = 4
        if not 'max_hamming' in kwargs:
            kwargs['min_hamming'] = 9
        if not 'weight' in kwargs:
            kwargs['weight'] = 10
        
        #get all single mutants of the seq
        single_muts, single_names = self.first_order_mutagenesis(start_seq)
        
        #get all scores
        single_scores = self.scorer.score_seqs(single_muts)
        #add penalties
        if hamming_penalty is not None:
            penalty = self.first_order_hamming_penalty(single_muts,
                                                          kwargs['min_hamming'],
                                                          kwargs['max_hamming'],
                                                          kwargs['weight'])
            single_scores += penalty
            
        P = np.exp(single_scores/T)
                
        #fix don't change certain positions
        if fixed_sites is not None:
            if self.fixed_idx is None:
                self.get_fixed_idxs(fixed_sites)
            P[self.fixed_idx] = 0
        if restricted_aa is not None:
            if self.fixed_aa_idxs is None:
                self.get_restricted_aa(restricted_aa)
            P[self.fixed_aa_idxs] = 0
        
        P = P/np.sum(P)
                
        #sample the next step
        idx = np.random.choice(np.arange(len(P)), p=P)
        new_seq = single_muts[idx]
        new_name = single_names[idx]
        new_score = single_scores[idx]
        
        seq_ii = int(re.search('\d+', new_name).group(0))
        seq_aa = re.search('[A-Z]', new_name).group(0)
        
        #get score of previous seq
        start_score = self.scorer.score_seqs([start_seq])[0]
        start_aa = start_seq[seq_ii-1]
        wt_aa = self.wt_seq[seq_ii-1]
        
        #if score is different update
        if new_score != start_score:
            return (new_seq, seq_ii, seq_aa, start_aa, wt_aa, start_score, new_score, single_scores)
        else:
            return (start_seq, seq_ii, start_aa, start_aa, wt_aa, start_score, start_score, single_scores)
        
def summarize_report(report, cdr_i_sites):

    startseq = report.iloc[0]['seq0']
    sum_row = [startseq]
    newseq, newscore = report.iloc[-1]['seq1'], report.iloc[-1]['new_score']
    bestseq = report[report['new_score'] == report['new_score'].max()].iloc[-1]['seq1']
    bestscore = report['new_score'].max()
    
    for s, sc in [(newseq, newscore), (bestseq, bestscore)]:
        hdist = hamming(startseq, s)
        mutnames = util.id_mut_from_seq(startseq, s)
        mutsites = [int(re.search('\d+', x).group(0)) for x in mutnames]
        cdr_muts = np.array([0,0,0])
        for i,arr in enumerate(cdr_i_sites):
            for m in mutsites:
                if m in arr:
                    cdr_muts[i] += 1
        sum_row.extend([s, hdist, sc,
                   cdr_muts[0], cdr_muts[1], cdr_muts[2],
                   ':'.join(mutnames)])
    
    return (sum_row)

class Logistic_Regression_Scorer():
    def __init__(self, model, data_importer, args):
        self.model = model
        self.data = data_importer
        self.args = deepcopy(args)
        self.args.no_wt_cols = True
        
    def score_seqs(self, seqs):
        #encode the seqs
        X = self.data.seq_to_OH(args=self.args, X_seq=seqs)
        y = self.model.predict_proba(X)[:,1]
        
        return (y)

class CNN_Scorer():
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
        
    def score_seqs(self, seqs):
        df = pd.DataFrame(seqs, columns=['seq'])
        dataset = OneHotArrayDataset(df,'seq',label=None)
        loader = torch.utils.data.DataLoader(dataset = dataset, batch_size = 128, shuffle = False)
        scores = []
        for seq_array, labels in loader:
            inputs = torch.reshape(seq_array,(seq_array.shape[0],seq_array.shape[2],seq_array.shape[1])).float().to(self.device)
            outputs = self.model(inputs).to(self.device)
            if outputs.shape[0] > 1:
                outputs = outputs.squeeze().tolist()
                scores.extend(outputs)
            else:
                outputs = outputs.squeeze().tolist()
                scores.append(outputs)
        return (scores)
        