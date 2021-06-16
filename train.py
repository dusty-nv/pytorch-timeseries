#!/usr/bin/env python3
# coding: utf-8

import os
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dataset import Dataset
from model import Model
    

parser = argparse.ArgumentParser()

parser.add_argument('--data', default='', type=str, required=True, help='path to CSV file')
parser.add_argument('--inputs', default='', type=str, required=True, help='name of input columns (comma-separated)')
parser.add_argument('--outputs', default='', type=str, required=True, help='name of output columns (comma-separated)')
parser.add_argument('--history', default=0, type=int, help='sequence history (in timesteps). GRU/RNN models should use history > 0, others should use history=0')
parser.add_argument('--horizon', default=1, type=int, help='forecasting horizon (in timesteps)')
parser.add_argument('--input-scaler', default='standard', choices=['none', 'minmax', 'standard'], help='dataset preprocessing scaler to use')
parser.add_argument('--output-scaler', default='standard', choices=['none', 'minmax', 'standard'], help='dataset preprocessing scaler to use')
parser.add_argument('--classification', action='store_true', help='set for classification datasets')

parser.add_argument('--model', default='linear', type=str, help='PyTorch checkpoint or model architecture: ' + ','.join(Model.available_models()))
parser.add_argument('--epochs', default=1000, type=int, help='number of training epochs')
parser.add_argument('--batch-size', default=-1, type=int, help='the batch size (default is entire dataset)')
parser.add_argument('--learning-rate', default=0.05, type=float, help='learning rate')
parser.add_argument('--scheduler', default='StepLR_250', type=str, help='learning rate scheduler')

parser.add_argument('--plot', default='', type=str, help='path to save plot (by default will be <data>.jpg)')
parser.add_argument('--plot-x', default='0', type=str, help='column name or index of the plot x-axis')
parser.add_argument('--plot-metrics', default='rmse,r2', help='error metrics to evaluate (comma-separated: rmse,mse,mae,max_error,r2)')
parser.add_argument('--plot-width', default=1920, type=int, help='plot width (in pixels)')
parser.add_argument('--plot-height', default=1080, type=int, help='plot height (in pixels)')

args = parser.parse_args()

# set default plot filename from data filename
# and attempt to parse plot x column as int 
if not args.plot:
    args.plot = os.path.splitext(args.data)[0] + '.jpg'
   
try:
    args.plot_x = int(args.plot_x)
except ValueError:
    pass
        
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
          
# eval plot
fig, axes = plt.subplots(len(dataset.subsets), len(dataset.outputs), squeeze=False)
    
for s, subset in enumerate(dataset.subsets):
    loss, outputs = model.eval(dataset[subset], return_outputs=True)
    df, metrics = dataset[subset].merge(outputs, return_metrics=args.plot_metrics)

    for metric in metrics:
        print(f"{f'{subset} {metric.upper()}:':<11s} {metrics[metric]}")
    
    for o, output in enumerate(dataset.outputs):
        df.plot(x=args.plot_x, y=[output, 'predicted_' + output], ax=axes[s,o],
                title=f"{subset} ({', '.join([f'{metric.upper()}={metrics[metric][o]:.5g}' for metric in metrics])})")

fig.set_size_inches(args.plot_width/100, args.plot_height/100)
fig.suptitle(f'{args.data} - {args.model} model ({args.epochs} epochs, history={args.history}, horizon={args.horizon})')
fig.savefig(args.plot) 
        
print(f'saved plot to {args.plot}')
