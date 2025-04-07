import torch
from torch import nn
from torch.utils.data import WeightedRandomSampler, Dataset

import pandas as pd
import sys
import argparse
import os, re
from tqdm import tqdm
from importlib import reload
import time as time

sys.path.append('/n/groups/marks/projects/binding_affinity/')
from modelling_functions import get_training_parser, data_importer, plot_roc_curve

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, r2_score
from scipy.stats import spearmanr
sys.path.append('/n/groups/marks/projects/binding_affinity/model_zoo')
from dataset_classes import OneHotArrayDataset
from supervised_model_classes import CNN, MLP, LSTM

import torch.nn as nn 
from sklearn.linear_model import LogisticRegression, Lasso, LinearRegression

from multiprocessing import Pool
from argparse import ArgumentParser

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print (device)
def train(model, train_loader, criterion, optimizer, epoch, num_epochs,
          metrics,
          batch_size, l1=True, l1_lambda=0.001):
    for i, (seq_array, labels) in enumerate(train_loader): 
        inputs = seq_array.float().to(device)
        labels = labels.reshape(-1,1).to(device)
        # Forward + Backward + Optimize 
        optimizer.zero_grad() 
        outputs = model(inputs)
        if l1:
            l1_regularization = torch.tensor(0).to(device)
            for param in model.parameters():
                l1_regularization += torch.norm(param, 1).long().to(device)
            loss = criterion(outputs, labels.float()) + l1_regularization * l1_lambda
        else:
            loss = criterion(outputs, labels.float()) 
        loss.backward() 
        optimizer.step() 
#         print(i)
#         print((i + 2)*batch_size)
        if (i + 2)*batch_size >= len(train_loader.dataset): 
            print('Epoch: [% d/% d], Step: [% d/% d], Loss: %.4f'
                  % (epoch + 1, num_epochs, i + 1, 
                     len(train_loader.dataset) // batch_size, loss.data),flush=True) 
            #record progress
            metrics.append(['Train', epoch, loss.item(), np.nan, np.nan])

def test(model, test_loader, criterion, task='classification'):
    correct = 0
    total = 0
    test_loss = 0
    scores = []
    all_labels = []
    for seq_array, labels in test_loader: 
        inputs = seq_array.float().to(device)
        labels = labels.reshape(-1,1).to(device)
        outputs = model(inputs).to(device)
        test_loss += labels.shape[0]*criterion(outputs, labels.float())
        total += labels.shape[0]
        outputs = outputs.squeeze().tolist()
        all_labels.extend(labels.tolist())
        scores.extend(outputs)

    if task == 'classification':
        #calculate ROC-AUC
        fpr,tpr,thr = roc_curve(all_labels,scores)
        rocauc = auc(fpr, tpr)
        #calculate PR-AUC
        precision, recall, thresholds = precision_recall_curve(all_labels, scores)
        prauc = auc(recall, precision)
        extra_metrics = [rocauc, prauc]
        print('Test Loss: {:.5f} | Test ROC-AUC: {:.3f} | Test PR-AUC: {:.3f}'.format(test_loss/total, rocauc, prauc),flush=True)
    if task == 'regression':
        #calculate R2
        r2 = r2_score(all_labels, scores)
        rho = spearmanr(all_labels, scores).correlation
        extra_metrics = [r2, rho]
        print('Test Loss: {:.5f} | Test R^2: {:.3f} | Test spearman: {:.3f}'.format(test_loss/total, r2, rho),flush=True)

    return (scores, test_loss/total, extra_metrics)

def get_latest_ckpt(directory):
    ckpt_nms = os.listdir(directory)
    ckpt_nms = [c for c in ckpt_nms if re.search('.tar', c)]
    ckpt_epochs = pd.Series(ckpt_nms).str.extract('\w+_epoch(\d+)').iloc[:,0].values.astype(int)
    best_epoch_idx = np.argmax(ckpt_epochs)
    return (ckpt_nms[best_epoch_idx])

def get_rocauc(self,X,Y):
    yscore = self.model.predict_proba(X)[:,1]
    fpr,tpr,thr = roc_curve(Y,yscore)
    return (auc(fpr, tpr))

global _extract_rep
def _extract_rep(pt_file):
    sel_layer = 33
    res = torch.load(pt_file)
    seq_ID = res['label']
    rep = res['mean_representations'][sel_layer].numpy()[np.newaxis, :] #1 x D matrix

    return (seq_ID, rep)
    
def import_reps(rep_dir):
    
    sum_file = os.path.join(rep_dir, 'all_reps.csv')
    labels_file = os.path.join(rep_dir, 'all_labels.csv')
    
    if not os.path.isfile(sum_file):
        pt_nms = [os.path.join(rep_dir,f) for f in os.listdir(rep_dir) if f.endswith('.pt')]

        print (f'Opening representations from {rep_dir}')
        ST = time.time()
        #res_ex = [_extract_rep(nm) for nm in tqdm(pt_nms)]
        with Pool() as p:
            res_ex = list(p.imap(_extract_rep, pt_nms))
        print ('Took {:.1f}s'.format(time.time()-ST))

        labels, rX = [], []
        for l,x in res_ex:
            labels.append(l)
            rX.append(x)

        labels = pd.DataFrame(labels, columns=['seq_ID']).reset_index().set_index('seq_ID')
        rX = np.vstack(rX)
        
        #save
        pd.DataFrame(rX).to_csv(sum_file, index=False)
        labels.to_csv(labels_file, index=True)
        
    else:
        print (f'Opening representations from {rep_dir} from file')
        ST = time.time()
        rX = pd.read_csv(sum_file).values
        labels = pd.read_csv(labels_file, index_col=0)
        print ('Took {:.1f}s'.format(time.time()-ST))
    
    return (rX, labels)


class NumpyDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]
    
    
class MLP(nn.Module):
    def __init__(self, input_size, dropout=0.25, n_hid=128):
        super().__init__()
        self.model = nn.Sequential(
            
            nn.Dropout(dropout),
            nn.Linear(input_size, n_hid), #input size is the seq length
            nn.ReLU(),
            nn.BatchNorm1d(n_hid),
            
            nn.Dropout(dropout),
            nn.Linear(n_hid, 1),
        )
    def forward(self, input_tensor):
        ##input tensor: (N, L, D)  (D: alphabet size, L: seq length)
        N = input_tensor.shape[0]
        return self.model(input_tensor.view(N, -1))
    

def main():
    
    parser = ArgumentParser()
    parser.add_argument('--proj')
    parser.add_argument('--X_fnm')
    parser.add_argument('--Y_fnm')
    parser.add_argument('--train_seqs')
    parser.add_argument('--test_seqs')
    parser.add_argument('--train_Y_col')
    parser.add_argument('--test_Y_col')
    parser.add_argument('--ckpt_dir')
    
    parser.add_argument('--num_epochs', default=100, type=int)
    parser.add_argument('--l1', action='store_true', default=True)
    parser.add_argument('--l1_lambda', default=0.001)
    args = parser.parse_args()
    
    ### Set modelling parameters
    args.balanced = True
    args.bs = 256
    args.random_seed = 0
    args.num_epochs = args.num_epochs
    args.lr = 0.0001
    args.l1 = True
    args.model = 'MLP'
    
    rep_parent = '/n/groups/marks/projects/binding_affinity/model_zoo/semi_supervised_ESM2'
    rep_dir = os.path.join(rep_parent, f'{args.proj}/{args.train_seqs}')
    test_rep_dir = os.path.join(rep_parent, f'{args.proj}/{args.test_seqs}')
    singles_rep_dir = os.path.join(rep_parent, f'{args.proj}/singles')
    
    X_fnm = args.X_fnm
    Y_fnm = args.Y_fnm
    train_seqs = args.train_seqs
    test_seqs = args.test_seqs
    train_Y_col = args.train_Y_col
    test_Y_col = args.test_Y_col
    ckpt_dir = args.ckpt_dir
    
    if 'logratio' in train_Y_col:
        criterion = nn.MSELoss()
        task = 'regression'
        metrics_cols = ['Train/Test', 'epoch', 'Loss', 'R2', 'spearman']
    elif 'enriched' in train_Y_col:
        criterion = nn.BCEWithLogitsLoss()
        task = 'classification'
        metrics_cols = ['Train/Test', 'epoch', 'Loss', 'ROC-AUC', 'PR-AUC']
    else:
        raise ValueError('Use either logratio (regression) or enriched (classification) Y cols')
    
    ### IMPORT THE ESM REPRESENTATIONS AND PREPARE THE DATA

    #split up into the train and val sets
    X_df = pd.read_csv(args.X_fnm).set_index('seq_ID')
    Y_df = pd.read_csv(args.Y_fnm).set_index('seq_ID')
    X_df.index = pd.Series(X_df.index).str.replace('\/', '_')
    Y_df.index = pd.Series(Y_df.index).str.replace('\/', '_')

    rX, labels = import_reps(rep_dir)
    
    Y_col = '_'.join(train_Y_col.split('_')[:2])

    train_idx = Y_df.index[Y_df[f'{Y_col}_split'] == 'train'].values
    train_idx = train_idx[np.isin(train_idx, labels.index)]
    train_Y = Y_df.loc[train_idx, train_Y_col].values.astype(int)
    train_X = rX[labels.loc[train_idx, 'index']]

    val_idx = Y_df.index[Y_df[f'{Y_col}_split'] == 'test'].values
    val_idx = val_idx[np.isin(val_idx, labels.index)]
    val_Y = Y_df.loc[val_idx, train_Y_col].values.astype(int)
    val_X = rX[labels.loc[val_idx, 'index']]
    
    if args.balanced & (task == 'classification'):
        train_ratio = np.bincount(train_Y)
        class_count = train_ratio.tolist()
        train_weights = 1./torch.tensor(class_count, dtype=torch.float)
        train_sampleweights = train_weights[train_Y]
        sampler = WeightedRandomSampler(weights=train_sampleweights, num_samples = len(train_sampleweights))
    else:
        sampler = None    


    train_dataset = NumpyDataset(train_X, train_Y)
    val_dataset = NumpyDataset(val_X, val_Y)

    #construct data loaders
    train_loader = torch.utils.data.DataLoader(dataset = train_dataset,  
                                               batch_size = args.bs,  
                                               shuffle = False,
                                               sampler = sampler,
                                               drop_last = True)
    val_loader = torch.utils.data.DataLoader(dataset = val_dataset,  
                                                  batch_size = args.bs,  
                                                  shuffle = False)
    
    
    ### TRAIN MODEL
    
    input_size = train_X.shape[1]
    model = MLP(input_size)
    model.to(device)

    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print ('.........MODEL INITIALIZED.............')
    #directory organizing
    ckpt_dir2 = os.path.join(ckpt_dir, 'ckpts')
    if not os.path.isdir(ckpt_dir2):
        os.makedirs(ckpt_dir2)
    #Train
    val_loss = 1e6
    logstep = 10
    metrics = []
    for epoch in range(args.num_epochs):
        #print ('Training epoch: {}'.format(epoch))
        model.train()
        train(model, train_loader, criterion, optimizer, epoch, 
            args.num_epochs, args.bs, args.l1, args.l1_lambda)
        model.eval()
        #get val performance
        if (epoch % logstep) == 0:
            ckpt_nm = os.path.join(ckpt_dir2, f'{args.model}_epoch{epoch}_W{args.l1_lambda}_ckpt.tar')
            _, loss, extra_metrics = test(model, val_loader, criterion, task=task)
            metrics.append(['Test', epoch, loss.item(), extra_metrics[0], extra_metrics[1]])
            if loss < val_loss:
                val_loss = loss
                torch.save(model.state_dict(),ckpt_nm)

    print ('.........MODEL TRAINED.............')
    
    m_df = pd.DataFrame(metrics, columns=metrics_cols)

    test_m_df = m_df[m_df['Train/Test'] == 'Test']
    test_m_dfm = pd.melt(test_m_df, id_vars=['Train/Test', 'epoch'],
                         var_name='metric', value_name='value')

    #save training figures
    train_fignm = os.path.join(ckpt_dir, 'val_stats.pdf')
    sns.set_theme(style='white', font_scale=1.5)
    f = sns.relplot(data=test_m_dfm, x='epoch', y='value',
                 col='metric', hue='metric', kind='line',
                    linewidth=2,
                facet_kws=dict(sharey=False),
               legend=False)
    f.fig.savefig(train_fignm, bbox_inches='tight')
    
    
    ### SAVE SCORES
    
    #load best ckpt
    best_ckpt = os.path.join(ckpt_dir2, get_latest_ckpt(ckpt_dir2))
    load_model = MLP(input_size)
    load_model.load_state_dict(torch.load(best_ckpt, map_location=device))

    #load all data

    alltrain_X, alltrain_labels = import_reps(rep_dir)
    alltrain_Y = Y_df.loc[alltrain_labels.index, train_Y_col]
    alltrain_X = torch.Tensor(alltrain_X)
    alltrain_labels[train_Y_col] = alltrain_Y

    alltest_X, alltest_labels = import_reps(test_rep_dir)
    alltest_X = torch.Tensor(alltest_X)
    alltest_Y = Y_df.loc[alltest_labels.index, test_Y_col]
    alltest_labels[test_Y_col] = alltest_Y

    singles_X, singles_labels = import_reps(singles_rep_dir)
    singles_X = torch.Tensor(singles_X)
    singles_Y = np.ones(len(singles_X))

    #save scores of the datasets
    alltrain_scores = load_model(alltrain_X).detach().numpy()
    alltrain_labels['score'] = alltrain_scores
    alltrain_labels.to_csv(os.path.join(ckpt_dir,
                            'F1_M_scores.csv'), index=True)

    alltest_scores = load_model(alltest_X).detach().numpy()
    alltest_labels['score'] = alltest_scores
    alltest_labels.to_csv(os.path.join(ckpt_dir,
                            'F2_M_scores.csv'), index=True)

    singles_scores= load_model(singles_X).detach().numpy()
    singles_labels['score'] = singles_scores
    singles_labels.to_csv(os.path.join(ckpt_dir,
                            'singles_scores.csv'), index=True)


    ### TEST PERFORMANCE

    if task == 'classification':
        #calculate ROC-AUC
        fpr,tpr,thr = roc_curve(alltest_labels[test_Y_col], alltest_scores)
        rocauc = auc(fpr, tpr)
        #calculate PR-AUC
        precision, recall, thresholds = precision_recall_curve(alltest_labels[test_Y_col], alltest_scores)
        prauc = auc(recall, precision)
        extra_metrics = [rocauc, prauc]
        print('Test ROC-AUC: {:.3f} | Test PR-AUC: {:.3f}'.format(rocauc, prauc),flush=True)
    if task == 'regression':
        #calculate R2
        r2 = r2_score(alltest_labels[test_Y_col], alltest_scores)
        rho = spearmanr(alltest_labels[test_Y_col], alltest_scores).correlation
        extra_metrics = [r2, rho]
        print('Test R^2: {:.3f} | Test spearman: {:.3f}'.format(r2, rho),flush=True)

        
if __name__ == '__main__':
    main()