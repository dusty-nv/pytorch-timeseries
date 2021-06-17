#!/usr/bin/env python3
# coding: utf-8

import os
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', default='data/weather_data/', type=str)
parser.add_argument('-o', '--output', default='data/weather.csv', type=str)
parser.add_argument('--city', default='Vancouver', type=str)
args = parser.parse_args()


df = None
attributes = ['temperature', 'humidity', 'pressure', 'wind_direction', 'wind_speed']

for attribute in attributes:
    df_attr = pd.read_csv(os.path.join(args.input, attribute + '.csv'), usecols=['datetime', args.city], index_col=0, parse_dates=[0])
    df_attr.rename(columns={args.city:attribute}, inplace=True)
    
    if df is None:
        df = df_attr
    else:
        df = pd.merge(df, df_attr, how='inner', left_index=True, right_index=True)

df.dropna(inplace=True)
        
# convert kelvin to fahrenheit
df['temperature'] = (df['temperature'] - 273.15) * (9/5) + 32.0
print(df)

df.to_csv(args.output)
print(f"saved {len(df)} rows to {args.output}")