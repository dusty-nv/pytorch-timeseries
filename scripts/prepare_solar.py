#!/usr/bin/env python3
# coding: utf-8

import os
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', default='data/solar_power', type=str)
parser.add_argument('-o', '--output', default='data/solar_power.csv', type=str)
parser.add_argument('--plant', default=1, type=int, help='Plant 1 or Plant 2')

args = parser.parse_args()


df_power = pd.read_csv(os.path.join(args.input, f'Plant_{args.plant}_Generation_Data.csv'), index_col=0, parse_dates=True, dayfirst=True)
df_weather = pd.read_csv(os.path.join(args.input, f'Plant_{args.plant}_Weather_Sensor_Data.csv'), index_col=0, parse_dates=True)

print(df_power)
print(df_weather)



# https://stackoverflow.com/a/38985129
df_power = df_power.groupby('DATE_TIME')['DC_POWER', 'AC_POWER'].sum()

print(df_power)

df_merged = pd.merge(df_power, df_weather, how='inner', left_index=True, right_index=True)
df_merged = df_merged[['DC_POWER', 'AC_POWER', 'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION']]

print(df_merged)

df_merged.dropna(inplace=True)
df_merged.to_csv(args.output)

print(f"saved {len(df_merged)} rows to {args.output}")