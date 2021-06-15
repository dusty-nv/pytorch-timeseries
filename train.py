#!/usr/bin/env python3
# coding: utf-8

import argparse

from dataset import Dataset
from model import Model
    

parser = argparse.ArgumentParser()

parser.add_argument('--data', default='', type=str, required=True, help='path to CSV file')
parser.add_argument('--inputs', default='', type=str, required=True, help='name of input columns (comma-separated)')
parser.add_argument('--outputs', default='', type=str, required=True, help='name of output columns (comma-separated)')
parser.add_argument('--history', default=4, type=int, help='sequence history (in timesteps)')
parser.add_argument('--horizon', default=1, type=int, help='forecasting horizon (in timesteps)')
parser.add_argument('--input-scaler', default='standard', choices=['none', 'minmax', 'standard'], help='dataset preprocessing scaler to use')
parser.add_argument('--output-scaler', default='standard', choices=['none', 'minmax', 'standard'], help='dataset preprocessing scaler to use')
parser.add_argument('--classification', action='store_true', help='set for classification datasets')

parser.add_argument('--model', default='gru', type=str, help='PyTorch checkpoint or model architecture: ' + ','.join(Model.available_models()))
parser.add_argument('--epochs', default=1000, type=int, help='number of training epochs')
parser.add_argument('--batch-size', default=-1, type=int, help='the batch size (default is entire dataset)')
parser.add_argument('--learning-rate', default=0.05, type=float, help='learning rate')
parser.add_argument('--scheduler', default='StepLR_250', type=str, help='learning rate scheduler')

args = parser.parse_args()
print(args)
    
    
# load dataset
dataset = Dataset(args.data, args.inputs, args.outputs,
                  input_scaler=args.input_scaler,
                  output_scaler=args.output_scaler,
                  classification=args.classification,
                  history=args.history, horizon=args.horizon)

print(dataset.df)

# create model
model = Model(args.model, dataset.num_inputs, dataset.num_outputs)

# train
model.train(dataset, epochs=args.epochs, batch_size=args.batch_size, 
            learning_rate=args.learning_rate, scheduler=args.scheduler)
          
# eval
loss, outputs = model.eval(dataset['val'], return_outputs=True)

print('val loss', loss)
print(outputs)
