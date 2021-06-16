#!/usr/bin/env python3
# coding: utf-8

import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', default='data/weather_data/temperature.csv', type=str)
parser.add_argument('-o', '--output', default='data/weather.csv', type=str)
parser.add_argument('--city', default='Vancouver', type=str)
args = parser.parse_args()


df = pd.read_csv(args.input, usecols=['datetime', args.city], parse_dates=[0])
#print(df)

df = df.dropna()
#print(df)

#df['day'] = df['datetime'].dt.dayofyear
#df['hour'] = df['datetime'].dt.hour

# convert kelvin to fahrenheit
df[args.city] = (df[args.city] - 273.15) * (9/5) + 32.0

df.rename(columns={args.city:'temperature'}, inplace=True)
print(df)

df.to_csv(args.output, index=False)
print(f"saved {len(df)} rows to {args.output}")