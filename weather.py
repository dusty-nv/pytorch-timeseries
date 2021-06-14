#!/usr/bin/env python3
# coding: utf-8

import argparse
import torch

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, MinMaxScaler


torch.manual_seed(1)  # reproducibility

  
parser = argparse.ArgumentParser()

parser.add_argument('--data', default='data/weather.csv', type=str)
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

df['day'] = df['datetime'].dt.dayofyear
df['hour'] = df['datetime'].dt.hour

df = df[['day', 'hour', 'temperature']]
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
    
def create_dataset(data, horizon):
    x = to_pytorch(data[:-horizon,:])
    y = to_pytorch(np.roll(data[:,-1],-horizon,axis=0)[:-horizon]).unsqueeze(dim=-1)
    return x, y

train_split = int(len(data) * args.split)

x_train, y_train = create_dataset(data[:train_split,:], args.horizon)
x_test, y_test = create_dataset(data[train_split:,:], args.horizon)
 
print('x_train', x_train.shape)
print('y_train', y_train.shape)

print('x_test', x_test.shape)
print('y_test', y_test.shape)


# create model
net = torch.nn.Sequential(
        torch.nn.Linear(x_train.shape[1], 200),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(200, 100),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(100, 1),
    ).cuda()


# create loss function and solver
criterion = torch.nn.MSELoss().cuda()
optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)  
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 250, 0.5)


# train
for epoch in range(args.epochs):
    net.train()
    
    y_pred = net(x_train)
    loss = criterion(y_pred, y_train)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    
    net.eval()
    
    with torch.no_grad():
        y_pred = net(x_test)
        test_loss = criterion(y_pred, y_test)
        print(f"Epoch {epoch:03d}  LR={scheduler.get_last_lr()[0]}  train_loss={loss:.8f}  test_loss={test_loss:.8f}")


# print out actual vs predicted values 
def unscale(array, resize=None):
    if resize: array = np.concatenate((np.zeros((resize[0], resize[1]-1)), array), axis=1)
    array = scaler.inverse_transform(array)
    return array[:,-1] if resize else array
    
x_test = to_numpy(x_test)
y_test = to_numpy(y_test)
y_pred = to_numpy(y_pred)
 
if scaler:
    x_test = unscale(x_test)
    y_test = unscale(y_test, x_test.shape)
    y_pred = unscale(y_pred, x_test.shape)

print('')
print('x_test', x_test)
print('y_test', y_test)
print('y_pred', y_pred)
