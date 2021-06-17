#!/usr/bin/env python3
# coding: utf-8

import os
import sys
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
parser.add_argument('--history', default=0, type=int, help='lookback sequence history (in timesteps). GRU/RNN models should use history > 0, others should use history=0')
parser.add_argument('--horizon', default=0, type=int, help='forecasting horizon - how far into the future to predict (in timesteps)')
parser.add_argument('--input-scaler', default='standard', choices=['none', 'minmax', 'standard'], help='dataset preprocessing scaler to use')
parser.add_argument('--output-scaler', default='standard', choices=['none', 'minmax', 'standard'], help='dataset preprocessing scaler to use')
parser.add_argument('--classification', action='store_true', help='set for classification datasets')

parser.add_argument('--model', default='linear', type=str, help='PyTorch checkpoint or model architecture: ' + ','.join(Model.available_models()))
parser.add_argument('--epochs', default=250, type=int, help='number of training epochs')
parser.add_argument('--batch-size', default=-1, type=int, help='the batch size (default is entire dataset)')
parser.add_argument('--learning-rate', default=0.05, type=float, help='learning rate')
parser.add_argument('--scheduler', default='StepLR_250', type=str, help='learning rate scheduler (StepLR_* or ReduceLROnPlateau_*)')
parser.add_argument('--metrics', default='RMSE,R2', help='error metrics to evaluate (comma-separated: rmse,mse,mae,max_error,r2,f1,accuracy,precision,recall)')

parser.add_argument('--plot', default='', type=str, help='path to save plot (by default will be <data>.jpg)')
parser.add_argument('--plot-x', default='0', type=str, help='column name or index of the plot x-axis')
parser.add_argument('--plot-width', default=1920, type=int, help='plot width (in pixels)')
parser.add_argument('--plot-height', default=1080, type=int, help='plot height (in pixels)')

args = parser.parse_args()

# set some default args for classification
if args.classification:
    if args.metrics == 'RMSE,R2':
        args.metrics = 'accuracy,precision,recall,F1'
    
    if args.output_scaler == 'standard':
        args.output_scaler = 'none'
        
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
          
# eval
metrics = {}

for subset in dataset.subsets:
    _, _, outputs = model.eval(dataset[subset], return_outputs=True)
    df, metrics[subset] = dataset[subset].merge(outputs, return_metrics=args.metrics)
    
    for metric in metrics[subset]:
        print(f"{f'{subset} {metric}:':<{len(max(metrics[subset].keys(), key=len)) + 7}s} {metrics[subset][metric]}")
        
    print("")
 
# plot
if args.classification:
    sys.exit()  # classification datasets aren't plotted
    
fig, axes = plt.subplots(len(dataset.subsets), len(dataset.outputs), squeeze=False, tight_layout=dict(rect=[0, 0, 1, 0.95]))
    
for s, subset in enumerate(dataset.subsets):
    for o, output in enumerate(dataset.outputs):
        df.plot(x=args.plot_x, y=[output, output + '^'], ax=axes[s,o], alpha=0.85,
                title=f"{subset} (N={len(dataset[subset])}, {', '.join([f'{metric}={metrics[subset][metric][o]:.5g}' for metric in metrics[subset]])})")
       
fig.set_size_inches(args.plot_width/100, args.plot_height/100)
fig.suptitle(f"{os.path.basename(args.data)} - {args.model} model ({args.epochs} epochs{f', history={args.history}' if args.history else ''}{f', horizon={args.horizon}' if args.horizon else ''})")
fig.savefig(args.plot) 
        
print(f'saved plot to {args.plot}')
