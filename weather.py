#!/usr/bin/env python3
# coding: utf-8

import math
import argparse
import torch
import torch.nn as nn

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, MinMaxScaler


torch.manual_seed(1)  # reproducibility

  
parser = argparse.ArgumentParser()

parser.add_argument('--data', default='data/weather.csv', type=str)
parser.add_argument('--history', default=8, type=int, help='sequence history (in hours)')
parser.add_argument('--horizon', default=1, type=int, help='forecasting horizon (in hours)')
parser.add_argument('--split', default=0.8, type=float, help='train/test dataset split')
parser.add_argument('--scaler', default='minmax', choices=['none', 'minmax', 'standard'], help='dataset preprocessing scaler to use')
parser.add_argument('--lr', '--learning-rate', default=0.05, type=float, help='learning rate')
parser.add_argument('--epochs', default=1000, type=int, help='number of training epochs')

args = parser.parse_args()
print(args)


# load data
print(f"loading {args.data}")
df = pd.read_csv(args.data, parse_dates=[0])

'''
df['day'] = df['datetime'].dt.dayofyear
df['hour'] = df['datetime'].dt.hour

df = df[['day', 'hour', 'temperature']]
'''

df = df[['temperature']]
print(df)


# pre-process data
if args.scaler == 'minmax':
    scaler = MinMaxScaler(feature_range=(-1, 1))
elif args.scaler == 'standard':
    scaler = StandardScaler()
else:
    scaler = None

if scaler:
    data = scaler.fit_transform(df.values)
else:
    data = df.values
    
print(data)
print(data.shape)

# create PyTorch datasets
def to_pytorch(array):
    return torch.from_numpy(array).type(torch.FloatTensor).cuda()

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    
def unscale(array, resize=None):
    if not scaler: return array
    if len(array.shape) == 0: array = array.reshape(-1, 1)
    if len(array.shape) == 1: array = np.expand_dims(array, 0)
    #if resize: array = np.concatenate((np.zeros((resize[0], resize[1]-1)), array), axis=1)
    array = scaler.inverse_transform(array)
    #return array[:,-1] if resize else array
    return array
    
def generate_sequences(data, sequence_length):
    if sequence_length == 1:
        return np.expand_dims(data,1)
    seq = []
    for index in range(len(data) - sequence_length): 
        seq.append(data[index : index + sequence_length]) 
    return np.array(seq)
            
def create_dataset(data, history, horizon):
    # shift the data by the forecast length
    x = data[:-horizon,:]
    y = np.roll(data[:,-1],-horizon,axis=0)[:-horizon]
    
    # generate sequences
    x = generate_sequences(x, history)
    y = generate_sequences(y, history)[:,-1]
    
    # cast to pytorch tensors
    x = to_pytorch(x)
    y = to_pytorch(y).unsqueeze(dim=-1)
    
    return x, y

train_split = int(len(data) * args.split)

x_train, y_train = create_dataset(data[:train_split,:], args.history, args.horizon)
x_test, y_test = create_dataset(data[train_split:,:], args.history, args.horizon)
 
print('x_train', x_train.shape)
print('y_train', y_train.shape)

print('x_test', x_test.shape)
print('y_test', y_test.shape)


# create model
class GRU(nn.Module):
    def __init__(self, input_dim=1, output_dim=1, hidden_dim=32, num_layers=2):
        super(GRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.0)
        self.fc1 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().cuda()
        # UserWarning: RNN module weights are not part of single contiguous chunk of memory
        #self.gru.flatten_parameters()   
        x, (hn) = self.gru(x, (h0.detach()))
        x = self.fc1(x[:, -1, :]) 
        return x

net = GRU(x_train.shape[-1], 1).cuda()

# create loss function and solver
criterion = torch.nn.MSELoss().cuda()
optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)  
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 250, 0.5)

def RMSE(y_pred, y):
    return math.sqrt((np.square(unscale(to_numpy(y_pred)) - unscale(to_numpy(y)))).mean(axis=0).item())
    
# train
for epoch in range(args.epochs):
    net.train()
    
    y_pred = net(x_train)
    train_loss = criterion(y_pred, y_train)
    train_rmse = RMSE(y_pred, y_train)
    
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()
    scheduler.step()
    
    net.eval()
    
    with torch.no_grad():
        y_pred = net(x_test)
        test_loss = criterion(y_pred, y_test)
        test_rmse = RMSE(y_pred, y_test)
        #unscaled_loss = unscale(np.array(loss.item())).item()
        #unscaled_test_loss = unscale(np.array(test_loss.item())).item()
        print(f"Epoch {epoch:03d}  LR={scheduler.get_last_lr()[0]}  train_loss={train_loss:.8f}  test_loss={test_loss:.8f}  train_rmse={train_rmse:.8f}  test_err={test_rmse:.8f}")


# print out actual vs predicted values     
#x_test = to_numpy(x_test)
y_test = to_numpy(y_test)
y_pred = to_numpy(y_pred)
 
if scaler:
    #x_test = unscale(x_test)
    y_test = unscale(y_test, x_test.shape)
    y_pred = unscale(y_pred, x_test.shape)

print('')
#print('x_test', x_test)
print('y_test', y_test)
print('y_pred', y_pred)
