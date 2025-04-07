import torch
from torch import nn
from torch.utils.data import WeightedRandomSampler
import pandas as pd

import sys
import argparse
import os

sys.path.append('/n/groups/marks/projects/binding_affinity/')
from modelling_functions import get_training_parser, data_importer

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, r2_score
from scipy.stats import spearmanr
sys.path.append('/n/groups/marks/projects/binding_affinity/model_zoo')
from dataset_classes import OneHotArrayDataset
from supervised_model_classes import CNN, MLP, LSTM
# from plotting_tools import roc_plot, score_hist, param_heatmap

import torch.nn as nn 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print (device)
def train(model, train_loader, criterion, optimizer, epoch, num_epochs, batch_size, l1=True, l1_lambda=0.001):
    for i, (seq_array, labels) in enumerate(train_loader): 
        inputs = torch.reshape(seq_array,(seq_array.shape[0],seq_array.shape[2],seq_array.shape[1])).float().to(device)
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

def test(model, test_loader, criterion, task='classification'):
    correct = 0
    total = 0
    test_loss = 0
    scores = []
    all_labels = []
    for seq_array, labels in test_loader: 
        inputs = torch.reshape(seq_array,(seq_array.shape[0],seq_array.shape[2],seq_array.shape[1])).float().to(device)
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
        print('Test Loss: {:.5f} | Test ROC-AUC: {:.3f} | Test PR-AUC: {:.3f}'.format(test_loss/total, rocauc, prauc),flush=True)
        
    if task == 'regression':
        #calculate R2
        r2 = r2_score(all_labels, scores)
        rho = spearmanr(all_labels, scores).correlation
        print('Test Loss: {:.5f} | Test R^2: {:.3f} | Test spearman: {:.3f}'.format(test_loss/total, r2, rho),flush=True)

    return (scores, test_loss/total)



def modelling_args():
    parser = get_training_parser()
    
    parser.add_argument("--test_Y_col", type=str,
                        default = 'FACS2_MACS_enriched',
                        help = 'Which metric to test on')
    
    parser.add_argument('--balanced', action='store_true',
                        default=True, 
                        help='whether to balance classes for training')
    
    parser.add_argument('--bs', type=int,
                        default=256,
                        help='input batch size for training')
    
    parser.add_argument('--random_seed', type=float,
                        default=0)
    
    parser.add_argument('--num_epochs', type=int,
                        default=100,
                        help='number of epochs to train (default: 14)')
    
    parser.add_argument('--lr', type=float,
                        default=0.0001,
                        help='learning rate')
    
    parser.add_argument('--l1', action='store_true',
                        default=True,
                        help = 'whether to use l1 regularization or not')
    
    args = parser.parse_args()
    return args


#get args
args = modelling_args()


#edit any args
if 'logratio' in args.Y_col:
    args.balanced = False
elif 'enriched' in args.Y_col:
    args.balanced = True
else:
    raise ValueError('Use either logratio (regression) or enriched (classification) Y cols')

#set save parameters
sample_name = f'Train_{args.Y_col}'
print (sample_name)
sample_dir = os.path.join(args.exp_dir, args.experiment_name, args.sub_experiment, sample_name)
if not os.path.isdir(sample_dir):
    os.makedirs(sample_dir)
    
ckpt_dir = os.path.join(sample_dir, 'ckpt')
if not os.path.isdir(ckpt_dir):
    os.makedirs(ckpt_dir)
    
test_dir = os.path.join(sample_dir, f'Test_{args.test_Y_col}')
if not os.path.isdir(test_dir):
    os.makedirs(test_dir)
    
    
#import the data
data = data_importer(args)



#make training and test sets
def get_sets(data, Xcol, Ycol, split):
    R = '_'.join(Ycol.split('_')[:-1])
    if split == 'any':
        ids = data.Y_df.index[~data.Y_df[f'{R}_split'].isna()]
    else:
        ids = data.Y_df.index[data.Y_df[f'{R}_split'] == split]
    Y = data.Y_df.loc[ids, Ycol]
    X = data.X_df.loc[ids, Xcol]
    df = pd.concat([X,Y], axis=1)
    return (df)

train_df = get_sets(data, args.CDR_choice, args.Y_col, 'train')
val_df = get_sets(data, args.CDR_choice, args.Y_col, 'test') #for validation use the test set from the train col
test_df = get_sets(data, args.CDR_choice, args.test_Y_col, 'any')

train_dataset = OneHotArrayDataset(train_df,args.CDR_choice,args.Y_col)
val_dataset = OneHotArrayDataset(val_df,args.CDR_choice,args.Y_col)
test_dataset = OneHotArrayDataset(test_df,args.CDR_choice,args.test_Y_col)

#balance the datasets if needed
if args.balanced:
    train_ratio = np.bincount(train_df[args.Y_col])
    class_count = train_ratio.tolist()
    train_weights = 1./torch.tensor(class_count, dtype=torch.float)
    train_sampleweights = train_weights[train_df[args.Y_col].values]
    sampler = WeightedRandomSampler(weights=train_sampleweights, num_samples = len(train_sampleweights))
else:
    sampler = None
    
#construct data loaders
train_loader = torch.utils.data.DataLoader(dataset = train_dataset,  
                                           batch_size = args.bs,  
                                           shuffle = False,
                                           sampler = sampler,
                                           drop_last = True)
val_loader = torch.utils.data.DataLoader(dataset = val_dataset,  
                                              batch_size = args.bs,  
                                              shuffle = False)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset,  
                                              batch_size = args.bs,  
                                              shuffle = False)



#Initialize model
input_size = train_df[args.CDR_choice].apply(len).unique()[0]
if args.model == 'CNN':
    model = CNN(input_size)
elif args.model == 'MLP':
    input_size = input_size*20
    model = MLP(input_size)
elif args.model == 'LSTM':
    model = LSTM(input_size)
else:
    raise ValueError('Please pass in a valid model type for this script')
    
model.to(device)
if 'logratio' in args.Y_col:
    criterion = nn.MSELoss()
    task = 'regression'
elif 'enriched' in args.Y_col:
    criterion = nn.BCEWithLogitsLoss()
    task = 'classification'
else:
    raise ValueError('Use either logratio (regression) or enriched (classification) Y cols')
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
print ('.........MODEL INITIALIZED.............')
#Train
val_loss = 1e6
logstep = args.num_epochs//100
for epoch in range(args.num_epochs):
    print ('Training epoch: {}'.format(epoch))
    model.train()
    train(model, train_loader, criterion, optimizer, epoch, 
        args.num_epochs, args.bs, args.l1, args.l1_lambda)
    model.eval()
    #get val performance
    if (epoch % logstep) == 0:
        ckpt_nm = os.path.join(ckpt_dir, f'{args.model}_epoch{epoch}_W{args.l1_lambda}_ckpt.tar')
        _, loss = test(model, val_loader, criterion, task=task)
        if loss < val_loss:
            val_loss = loss
            torch.save(model.state_dict(),ckpt_nm)
            
print ('.........MODEL TRAINED.............')
#Test performance
final_test_scores, _ = test(model, test_loader, criterion, task=task)
test_df['score'] = final_test_scores
test_dir = os.path.join(sample_dir, f'Test_{args.test_Y_col}')
if not os.path.isdir(test_dir):
    os.makedirs(test_dir)

test_df.to_csv(os.path.join(test_dir, 'Test_scores.csv'))