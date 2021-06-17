#!/usr/bin/env python3
# coding: utf-8

import os
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', default='data/shuttle', type=str)
parser.add_argument('-o', '--output', default='data/shuttle.csv', type=str)
args = parser.parse_args()


# define column names
column_names = [f'{i}' for i in range(9)] + ['class']

# load data
df_train = pd.read_csv(os.path.join(args.input, 'shuttle.trn'), sep='\s+', header=None, names=column_names)
df_test = pd.read_csv(os.path.join(args.input, 'shuttle.tst'), sep='\s+', header=None, names=column_names)

print(df_train)
print(df_test)

# merge data
df = df_train.append(df_test, ignore_index=True)
df['class'] = df['class'] - 1   # rebase classes so they start with 0

print(df)

# show class stats
classes = df.groupby('class').groups
print('num classes', len(classes))

for cls in classes:
    print(f'class {cls} - {len(classes[cls])} samples')
    
# save output
df.to_csv(args.output, index=False)
print(f"saved {len(df)} rows to {args.output}")
