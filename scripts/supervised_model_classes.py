import torch
from torch import nn
import os, sys
import pandas as pd
import numpy as np

def get_latest_ckpt(directory):
    ckpt_nms = os.listdir(directory)
    ckpt_nms = [c for c in ckpt_nms if re.search('.tar', c)]
    ckpt_epochs = pd.Series(ckpt_nms).str.extract('\w+_epoch(\d+)').iloc[:,0].values.astype(int)
    best_epoch_idx = np.argmax(ckpt_epochs)
    return (ckpt_nms[best_epoch_idx])

class CNN(torch.nn.Module):
    def __init__(self,input_size):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=20,out_channels=32,kernel_size=3,stride=1),
            nn.BatchNorm1d(32),
            nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=32,out_channels=64,kernel_size=3,stride=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(3,stride=3))
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=64,out_channels=128,kernel_size=3,stride=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(3,stride=1))
        self.input_size = input_size
        self.linear_size = int(((self.input_size-7)/3)-3)*128
        self.linear = nn.Linear(self.linear_size, 1, bias=False) # 1024 is 128 channels * 8 width
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out.view(x.shape[0],out.shape[1]*out.shape[2]).to(self.device)
        out = self.linear(out)
        return out
    
    
    
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, num_classes=1):
        super(LSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        # self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        # -> x needs to be: (batch_size, seq, input_size)
        
        # or:
        #self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Set initial hidden states (and cell states for LSTM)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device) 
        
        # x: (n, 28, 28), h0: (2, n, 128)
        
        # Forward propagate RNN
        #out, _ = self.rnn(x, h0)  
        # or:
        out, _ = self.lstm(x, (h0,c0))  
        
        # out: tensor of shape (batch_size, seq_length, hidden_size)
        # out: (n, 28, 128)
        
        # Decode the hidden state of the last time step
        out = out[:, -1, :]
        # out: (n, 128)
         
        out = self.fc(out)
        # out: (n, 10)
        
        # out = self.sigmoid(out)
        return out

    
class MLP(nn.Module):
    def __init__(self, input_size, dropout=0.25, n_hid=64):
        super().__init__()
        self.model = nn.Sequential(
            
            nn.Dropout(dropout),
            nn.Linear(input_size, n_hid), #input size is the seq length
            nn.ReLU(),
            nn.BatchNorm1d(n_hid),
            
            nn.Dropout(dropout),            
            nn.Linear(n_hid, n_hid // 2),
            nn.ReLU(),
            nn.BatchNorm1d(n_hid // 2),
            
            nn.Dropout(dropout),
            nn.Linear(n_hid // 2, n_hid // 4),
            nn.ReLU(),
            nn.BatchNorm1d(n_hid // 4),
            
            nn.Dropout(dropout),
            nn.Linear(n_hid // 4, 1),
        )
    def forward(self, input_tensor):
        ##input tensor: (N, L, D)  (D: alphabet size, L: seq length)
        N = input_tensor.shape[0]
        return self.model(input_tensor.view(N, -1))