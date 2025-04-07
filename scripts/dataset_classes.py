import sys
import pickle
import math
import time
import pandas as pd
from textwrap import wrap
import json
import numpy as np
import os
import glob
import re
import sys
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import  Dataset

class OneHotArrayDataset(Dataset):
    def __init__(self, df, seq_col, label):
        self.samples = []
        k = SequencesToOneHot()
        arr = k.seqs_to_arr(df, seq_col = seq_col)
        #print('one hot shape',arr.shape)
        if label is not None:
            for x, y in zip(arr, df[label]):
                self.samples.append((x, y))
        else:
            for x in arr:
                self.samples.append((x, 0))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
    
class NonAlignedOneHotArrayDataset(Dataset):
    def __init__(self, df, seq_col, label, max_len_from_df):
        self.samples = []
        k = SequencesToOneHot_nonaligned()
        arr = k.seqs_to_arr(df, max_len = max_len_from_df, seq_col = seq_col)
        for x, y in zip(arr, df[label]):
            self.samples.append((x, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
    
class SequencesToOneHot_nonaligned():
    def __init__(self, alphabet='protein'):
        if alphabet == 'protein':
            self.aa_list = 'ACDEFGHIKLMNPQRSTVWY'
        else:
            self.aa_list = 'ACGT'
        self.aa_dict = {}
        for i,aa in enumerate(self.aa_list):
            self.aa_dict[aa] = i
    def one_hot_3D(self, s):
        x = np.zeros((len(s), len(self.aa_list)))
        for i, letter in enumerate(s):
            if letter in self.aa_dict:
                x[i , self.aa_dict[letter]] = 1
        return x
    def seqs_to_arr(self, df, max_len = None, seq_col = None):
        onehot_array = np.empty((len(df[seq_col]), max_len, 20))
        for s, seq in enumerate(df[seq_col].values):
            if type(seq)==float:
                seq = ''
            new_seq = seq.upper() + '-'*(max_len-len(seq))
            onehot_array[s] = self.one_hot_3D(new_seq)
        return onehot_array

class SequencesToOneHot():
    def __init__(self, alphabet='protein'):
        if alphabet == 'protein':
            self.aa_list = 'ACDEFGHIKLMNPQRSTVWY'
        else:
            self.aa_list = 'ACGT'
        self.aa_dict = {}
        for i,aa in enumerate(self.aa_list):
            self.aa_dict[aa] = i
    def one_hot_3D(self, s):
        x = np.zeros((len(s), len(self.aa_list)))
        for i, letter in enumerate(s):
            if letter in self.aa_dict:
                x[i , self.aa_dict[letter]] = 1
        return x
    def seqs_to_arr(self, df, seq_col=None):
        onehot_array = np.empty((len(df[seq_col]),len(df.iloc[0].loc[seq_col]),20))

        for s, seq in enumerate(df[seq_col].values):
            if type(seq)==float:
                seq = ''
            seq = seq.upper()
            onehot_array[s] = self.one_hot_3D(seq)
        return onehot_array