# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 12:25:13 2021

@author: 17405
to-do: streamline the data formatting phase: easily picking which variables I
want to forecast. Then, turn it into a local function or put it in a module.
"""
import math
import pickle
import numpy as np
import pandas as pd
from datetime import date, time, datetime, timedelta

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
myFmt = DateFormatter('%H:%M')

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# %%
dataset = pickle.load(open('dataset_unformatted.pkl', 'rb'))

log10_cn2_df = dataset['log10_cn2']
log10_cn2_np = log10_cn2_df.to_numpy().copy()

# dts = dataset.index.to_pydatetime()
dts = dataset.index
idx_night = np.logical_or(dts.hour.to_numpy() <= 7,
                          dts.hour.to_numpy() >= 21)
idx_night = np.logical_and(idx_night,
                           np.isnan(log10_cn2_np))
log10_cn2_tmp = log10_cn2_np.copy()
log10_cn2_tmp[idx_night] = -15
dataset['log10_cn2_night'] = log10_cn2_tmp

# %%
sequence_length = 8
forecast_length = 8
sequences_04hr_df = []
forecasts_04hr_df = []
for ii in range(len(dataset) - sequence_length - forecast_length):
    seq_slice = slice(ii, ii + sequence_length)
    this_sequence = dataset[
        ['temp', 'press', 'rh', 'wind_spd',
         'solar_irr', 'log10_cn2_night']].iloc[seq_slice]
    
    forecast_slice = slice(
        ii + sequence_length,
        ii + sequence_length + forecast_length)
    this_forecast = dataset['log10_cn2'].iloc[forecast_slice]
    
    condition1 = this_sequence.isnull().any().any()==False
    condition2 = this_forecast.isnull().any().any()==False
    if condition1 & condition2:
        sequences_04hr_df.append(this_sequence)
        forecasts_04hr_df.append(this_forecast)

sequences_04hr_np = np.array([x.to_numpy() for x in sequences_04hr_df])
forecasts_04hr_np = np.array([x.to_numpy() for x in forecasts_04hr_df])
sequences_04hr_first_dts = np.array([x.index[0].to_pydatetime() for x in sequences_04hr_df])
forecasts_04hr_first_dts = np.array([x.index[0].to_pydatetime() for x in forecasts_04hr_df])


sequence_length = 12
forecast_length = 8
sequences_06hr_df = []
forecasts_06hr_df = []
for ii in range(len(dataset) - sequence_length - forecast_length):
    seq_slice = slice(ii, ii + sequence_length)
    this_sequence = dataset[
        ['temp', 'press', 'rh', 'wind_spd',
         'solar_irr', 'log10_cn2_night']].iloc[seq_slice]
    
    forecast_slice = slice(
        ii + sequence_length,
        ii + sequence_length + forecast_length)
    this_forecast = dataset['log10_cn2'].iloc[forecast_slice]
    
    condition1 = this_sequence.isnull().any().any()==False
    condition2 = this_forecast.isnull().any().any()==False
    if condition1 & condition2:
        sequences_06hr_df.append(this_sequence)
        forecasts_06hr_df.append(this_forecast)

sequences_06hr_np = np.array([x.to_numpy() for x in sequences_06hr_df])
forecasts_06hr_np = np.array([x.to_numpy() for x in forecasts_06hr_df])
sequences_06hr_first_dts = np.array([x.index[0].to_pydatetime() for x in sequences_06hr_df])
forecasts_06hr_first_dts = np.array([x.index[0].to_pydatetime() for x in forecasts_06hr_df])


sequence_length = 16
forecast_length = 8
sequences_08hr_df = []
forecasts_08hr_df = []
for ii in range(len(dataset) - sequence_length - forecast_length):
    seq_slice = slice(ii, ii + sequence_length)
    this_sequence = dataset[
        ['temp', 'press', 'rh', 'wind_spd',
         'solar_irr', 'log10_cn2_night']].iloc[seq_slice]
    
    forecast_slice = slice(
        ii + sequence_length,
        ii + sequence_length + forecast_length)
    this_forecast = dataset['log10_cn2'].iloc[forecast_slice]
    
    condition1 = this_sequence.isnull().any().any()==False
    condition2 = this_forecast.isnull().any().any()==False
    if condition1 & condition2:
        sequences_08hr_df.append(this_sequence)
        forecasts_08hr_df.append(this_forecast)

sequences_08hr_np = np.array([x.to_numpy() for x in sequences_08hr_df])
forecasts_08hr_np = np.array([x.to_numpy() for x in forecasts_08hr_df])
sequences_08hr_first_dts = np.array([x.index[0].to_pydatetime() for x in sequences_08hr_df])
forecasts_08hr_first_dts = np.array([x.index[0].to_pydatetime() for x in forecasts_08hr_df])


sequence_length = 20
forecast_length = 8
sequences_10hr_df = []
forecasts_10hr_df = []
for ii in range(len(dataset) - sequence_length - forecast_length):
    seq_slice = slice(ii, ii + sequence_length)
    this_sequence = dataset[
        ['temp', 'press', 'rh', 'wind_spd',
         'solar_irr', 'log10_cn2_night']].iloc[seq_slice]
    
    forecast_slice = slice(
        ii + sequence_length,
        ii + sequence_length + forecast_length)
    this_forecast = dataset['log10_cn2'].iloc[forecast_slice]
    
    condition1 = this_sequence.isnull().any().any()==False
    condition2 = this_forecast.isnull().any().any()==False
    if condition1 & condition2:
        sequences_10hr_df.append(this_sequence)
        forecasts_10hr_df.append(this_forecast)

sequences_10hr_np = np.array([x.to_numpy() for x in sequences_10hr_df])
forecasts_10hr_np = np.array([x.to_numpy() for x in forecasts_10hr_df])
sequences_10hr_first_dts = np.array([x.index[0].to_pydatetime() for x in sequences_10hr_df])
forecasts_10hr_first_dts = np.array([x.index[0].to_pydatetime() for x in forecasts_10hr_df])


sequence_length = 24
forecast_length = 8
sequences_12hr_df = []
forecasts_12hr_df = []
for ii in range(len(dataset) - sequence_length - forecast_length):
    seq_slice = slice(ii, ii + sequence_length)
    this_sequence = dataset[
        ['temp', 'press', 'rh', 'wind_spd',
         'solar_irr', 'log10_cn2_night']].iloc[seq_slice]
    
    forecast_slice = slice(
        ii + sequence_length,
        ii + sequence_length + forecast_length)
    this_forecast = dataset['log10_cn2'].iloc[forecast_slice]
    
    condition1 = this_sequence.isnull().any().any()==False
    condition2 = this_forecast.isnull().any().any()==False
    if condition1 & condition2:
        sequences_12hr_df.append(this_sequence)
        forecasts_12hr_df.append(this_forecast)

sequences_12hr_np = np.array([x.to_numpy() for x in sequences_12hr_df])
forecasts_12hr_np = np.array([x.to_numpy() for x in forecasts_12hr_df])
sequences_12hr_first_dts = np.array([x.index[0].to_pydatetime() for x in sequences_12hr_df])
forecasts_12hr_first_dts = np.array([x.index[0].to_pydatetime() for x in forecasts_12hr_df])


sequence_length = 28
forecast_length = 8
sequences_14hr_df = []
forecasts_14hr_df = []
for ii in range(len(dataset) - sequence_length - forecast_length):
    seq_slice = slice(ii, ii + sequence_length)
    this_sequence = dataset[
        ['temp', 'press', 'rh', 'wind_spd',
         'solar_irr', 'log10_cn2_night']].iloc[seq_slice]
    
    forecast_slice = slice(
        ii + sequence_length,
        ii + sequence_length + forecast_length)
    this_forecast = dataset['log10_cn2'].iloc[forecast_slice]
    
    condition1 = this_sequence.isnull().any().any()==False
    condition2 = this_forecast.isnull().any().any()==False
    if condition1 & condition2:
        sequences_14hr_df.append(this_sequence)
        forecasts_14hr_df.append(this_forecast)

sequences_14hr_np = np.array([x.to_numpy() for x in sequences_14hr_df])
forecasts_14hr_np = np.array([x.to_numpy() for x in forecasts_14hr_df])
sequences_14hr_first_dts = np.array([x.index[0].to_pydatetime() for x in sequences_14hr_df])
forecasts_14hr_first_dts = np.array([x.index[0].to_pydatetime() for x in forecasts_14hr_df])


sequence_length = 32
forecast_length = 8
sequences_16hr_df = []
forecasts_16hr_df = []
for ii in range(len(dataset) - sequence_length - forecast_length):
    seq_slice = slice(ii, ii + sequence_length)
    this_sequence = dataset[
        ['temp', 'press', 'rh', 'wind_spd',
         'solar_irr', 'log10_cn2_night']].iloc[seq_slice]
    
    forecast_slice = slice(
        ii + sequence_length,
        ii + sequence_length + forecast_length)
    this_forecast = dataset['log10_cn2'].iloc[forecast_slice]
    
    condition1 = this_sequence.isnull().any().any()==False
    condition2 = this_forecast.isnull().any().any()==False
    if condition1 & condition2:
        sequences_16hr_df.append(this_sequence)
        forecasts_16hr_df.append(this_forecast)

sequences_16hr_np = np.array([x.to_numpy() for x in sequences_16hr_df])
forecasts_16hr_np = np.array([x.to_numpy() for x in forecasts_16hr_df])
sequences_16hr_first_dts = np.array([x.index[0].to_pydatetime() for x in sequences_16hr_df])
forecasts_16hr_first_dts = np.array([x.index[0].to_pydatetime() for x in forecasts_16hr_df])

# get the common forecasts across all input sequence lengths
set_04hr = set(forecasts_04hr_first_dts)
set_06hr = set(forecasts_06hr_first_dts)
set_08hr = set(forecasts_08hr_first_dts)
set_10hr = set(forecasts_10hr_first_dts)
set_12hr = set(forecasts_12hr_first_dts)
set_14hr = set(forecasts_14hr_first_dts)
set_16hr = set(forecasts_16hr_first_dts)
f = (pd.Series(list(set_04hr & set_06hr & set_08hr & set_10hr & set_12hr & set_14hr & set_16hr))\
     .sort_values()).dt.to_pydatetime()
idx_04hr = np.array([i for i, x in enumerate(forecasts_04hr_first_dts) if x in f])
idx_06hr = np.array([i for i, x in enumerate(forecasts_06hr_first_dts) if x in f])
idx_08hr = np.array([i for i, x in enumerate(forecasts_08hr_first_dts) if x in f])
idx_10hr = np.array([i for i, x in enumerate(forecasts_10hr_first_dts) if x in f])
idx_12hr = np.array([i for i, x in enumerate(forecasts_12hr_first_dts) if x in f])
idx_14hr = np.array([i for i, x in enumerate(forecasts_14hr_first_dts) if x in f])
idx_16hr = np.array([i for i, x in enumerate(forecasts_16hr_first_dts) if x in f])

sequences_04hr_np = sequences_04hr_np[idx_04hr]
sequences_04hr_df = [sequences_04hr_df[i] for i in idx_04hr]
forecasts_04hr_np = forecasts_04hr_np[idx_04hr]
forecasts_04hr_df = [forecasts_04hr_df[i] for i in idx_04hr]
sequences_04hr_first_dts = sequences_04hr_first_dts[idx_04hr]
forecasts_04hr_first_dts = forecasts_04hr_first_dts[idx_04hr]

sequences_06hr_np = sequences_06hr_np[idx_06hr]
sequences_06hr_df = [sequences_06hr_df[i] for i in idx_06hr]
forecasts_06hr_np = forecasts_06hr_np[idx_06hr]
forecasts_06hr_df = [forecasts_06hr_df[i] for i in idx_06hr]
sequences_06hr_first_dts = sequences_06hr_first_dts[idx_06hr]
forecasts_06hr_first_dts = forecasts_06hr_first_dts[idx_06hr]

sequences_08hr_np = sequences_08hr_np[idx_08hr]
sequences_08hr_df = [sequences_08hr_df[i] for i in idx_08hr]
forecasts_08hr_np = forecasts_08hr_np[idx_08hr]
forecasts_08hr_df = [forecasts_08hr_df[i] for i in idx_08hr]
sequences_08hr_first_dts = sequences_08hr_first_dts[idx_08hr]
forecasts_08hr_first_dts = forecasts_08hr_first_dts[idx_08hr]

sequences_10hr_np = sequences_10hr_np[idx_10hr]
sequences_10hr_df = [sequences_10hr_df[i] for i in idx_10hr]
forecasts_10hr_np = forecasts_10hr_np[idx_10hr]
forecasts_10hr_df = [forecasts_10hr_df[i] for i in idx_10hr]
sequences_10hr_first_dts = sequences_10hr_first_dts[idx_10hr]
forecasts_10hr_first_dts = forecasts_10hr_first_dts[idx_10hr]

sequences_12hr_np = sequences_12hr_np[idx_12hr]
sequences_12hr_df = [sequences_12hr_df[i] for i in idx_12hr]
forecasts_12hr_np = forecasts_12hr_np[idx_12hr]
forecasts_12hr_df = [forecasts_12hr_df[i] for i in idx_12hr]
sequences_12hr_first_dts = sequences_12hr_first_dts[idx_12hr]
forecasts_12hr_first_dts = forecasts_12hr_first_dts[idx_12hr]

sequences_14hr_np = sequences_14hr_np[idx_14hr]
sequences_14hr_df = [sequences_14hr_df[i] for i in idx_14hr]
forecasts_14hr_np = forecasts_14hr_np[idx_14hr]
forecasts_14hr_df = [forecasts_14hr_df[i] for i in idx_14hr]
sequences_14hr_first_dts = sequences_14hr_first_dts[idx_14hr]
forecasts_14hr_first_dts = forecasts_14hr_first_dts[idx_14hr]

sequences_16hr_np = sequences_16hr_np[idx_16hr]
sequences_16hr_df = [sequences_16hr_df[i] for i in idx_16hr]
forecasts_16hr_np = forecasts_16hr_np[idx_16hr]
forecasts_16hr_df = [forecasts_16hr_df[i] for i in idx_16hr]
sequences_16hr_first_dts = sequences_16hr_first_dts[idx_16hr]
forecasts_16hr_first_dts = forecasts_16hr_first_dts[idx_16hr]

# all the forecasts are equal now
forecasts_df = forecasts_16hr_df
forecasts_np = forecasts_16hr_np
forecasts_first_dts = forecasts_16hr_first_dts

# %% parse into train/validation/test datasets
train_dt_start = datetime(2020, 6, 1, 0, 0, 0)
train_dt_end = datetime(2020, 7, 18, 0, 0, 0)
valid_dt_start = train_dt_end
valid_dt_end = datetime(2020, 8, 2, 0, 0, 0)
test_dt_start = valid_dt_end
test_dt_end = datetime(2020, 8, 11, 0, 0, 0)

train_idx = np.logical_and(forecasts_first_dts >= train_dt_start,
                            forecasts_first_dts < train_dt_end)
valid_idx = np.logical_and(forecasts_first_dts >= valid_dt_start,
                            forecasts_first_dts < valid_dt_end)
test_idx = np.logical_and(forecasts_first_dts >= test_dt_start,
                          forecasts_first_dts < test_dt_end)

sequences_04hr_train_df = [x for (x,y) in zip(sequences_04hr_df, train_idx) if y]
sequences_06hr_train_df = [x for (x,y) in zip(sequences_06hr_df, train_idx) if y]
sequences_08hr_train_df = [x for (x,y) in zip(sequences_08hr_df, train_idx) if y]
sequences_10hr_train_df = [x for (x,y) in zip(sequences_10hr_df, train_idx) if y]
sequences_12hr_train_df = [x for (x,y) in zip(sequences_12hr_df, train_idx) if y]
sequences_14hr_train_df = [x for (x,y) in zip(sequences_14hr_df, train_idx) if y]
sequences_16hr_train_df = [x for (x,y) in zip(sequences_16hr_df, train_idx) if y]

sequences_04hr_train_np = sequences_04hr_np[train_idx]
sequences_06hr_train_np = sequences_06hr_np[train_idx]
sequences_08hr_train_np = sequences_08hr_np[train_idx]
sequences_10hr_train_np = sequences_10hr_np[train_idx]
sequences_12hr_train_np = sequences_12hr_np[train_idx]
sequences_14hr_train_np = sequences_14hr_np[train_idx]
sequences_16hr_train_np = sequences_16hr_np[train_idx]

forecasts_train_df = [x for (x,y) in zip(forecasts_df, train_idx) if y]
forecasts_train_np = forecasts_np[train_idx]
forecasts_first_dts_train = forecasts_first_dts[train_idx]


sequences_04hr_valid_df = [x for (x,y) in zip(sequences_04hr_df, valid_idx) if y]
sequences_06hr_valid_df = [x for (x,y) in zip(sequences_06hr_df, valid_idx) if y]
sequences_08hr_valid_df = [x for (x,y) in zip(sequences_08hr_df, valid_idx) if y]
sequences_10hr_valid_df = [x for (x,y) in zip(sequences_10hr_df, valid_idx) if y]
sequences_12hr_valid_df = [x for (x,y) in zip(sequences_12hr_df, valid_idx) if y]
sequences_14hr_valid_df = [x for (x,y) in zip(sequences_14hr_df, valid_idx) if y]
sequences_16hr_valid_df = [x for (x,y) in zip(sequences_16hr_df, valid_idx) if y]

sequences_04hr_valid_np = sequences_04hr_np[valid_idx]
sequences_06hr_valid_np = sequences_06hr_np[valid_idx]
sequences_08hr_valid_np = sequences_08hr_np[valid_idx]
sequences_10hr_valid_np = sequences_10hr_np[valid_idx]
sequences_12hr_valid_np = sequences_12hr_np[valid_idx]
sequences_14hr_valid_np = sequences_14hr_np[valid_idx]
sequences_16hr_valid_np = sequences_16hr_np[valid_idx]

forecasts_valid_df = [x for (x,y) in zip(forecasts_df, valid_idx) if y]
forecasts_valid_np = forecasts_np[valid_idx]
forecasts_first_dts_valid = forecasts_first_dts[valid_idx]


sequences_04hr_test_df = [x for (x,y) in zip(sequences_04hr_df, test_idx) if y]
sequences_06hr_test_df = [x for (x,y) in zip(sequences_06hr_df, test_idx) if y]
sequences_08hr_test_df = [x for (x,y) in zip(sequences_08hr_df, test_idx) if y]
sequences_10hr_test_df = [x for (x,y) in zip(sequences_10hr_df, test_idx) if y]
sequences_12hr_test_df = [x for (x,y) in zip(sequences_12hr_df, test_idx) if y]
sequences_14hr_test_df = [x for (x,y) in zip(sequences_14hr_df, test_idx) if y]
sequences_16hr_test_df = [x for (x,y) in zip(sequences_16hr_df, test_idx) if y]

sequences_04hr_test_np = sequences_04hr_np[test_idx]
sequences_06hr_test_np = sequences_06hr_np[test_idx]
sequences_08hr_test_np = sequences_08hr_np[test_idx]
sequences_10hr_test_np = sequences_10hr_np[test_idx]
sequences_12hr_test_np = sequences_12hr_np[test_idx]
sequences_14hr_test_np = sequences_14hr_np[test_idx]
sequences_16hr_test_np = sequences_16hr_np[test_idx]

forecasts_test_df = [x for (x,y) in zip(forecasts_df, test_idx) if y]
forecasts_test_np = forecasts_np[test_idx]
forecasts_first_dts_test = forecasts_first_dts[test_idx]

# %% get training min and max values
forecasts_log10_cn2_flat = forecasts_train_np.flatten()
sequences_temp_flat = sequences_16hr_train_np[:,:,0].flatten()
sequences_press_flat = sequences_16hr_train_np[:,:,1].flatten()
sequences_rh_flat = sequences_16hr_train_np[:,:,2].flatten()
sequences_wind_spd_flat = sequences_16hr_train_np[:,:,3].flatten()
sequences_solar_irr_flat = sequences_16hr_train_np[:,:,4].flatten()
sequences_log10_cn2_flat = sequences_16hr_train_np[:,:,5].flatten()

forecasts_train_min = forecasts_log10_cn2_flat.min()
forecasts_train_max = forecasts_log10_cn2_flat.max()

sequences_train_min = np.array(
    [sequences_temp_flat.min(), sequences_press_flat.min(),
      sequences_rh_flat.min(), sequences_wind_spd_flat.min(),
      sequences_solar_irr_flat.min(), sequences_log10_cn2_flat.min()])
sequences_train_max = np.array(
    [sequences_temp_flat.max(), sequences_press_flat.max(),
      sequences_rh_flat.max(), sequences_wind_spd_flat.max(),
      sequences_solar_irr_flat.max(),
      sequences_log10_cn2_flat.max()])

d_save = {
    "sequences_04hr_train_np": sequences_04hr_train_np,
    "sequences_06hr_train_np": sequences_06hr_train_np,
    "sequences_08hr_train_np": sequences_08hr_train_np,
    "sequences_10hr_train_np": sequences_10hr_train_np,
    "sequences_12hr_train_np": sequences_12hr_train_np,
    "sequences_14hr_train_np": sequences_14hr_train_np,
    "sequences_16hr_train_np": sequences_16hr_train_np,
    "sequences_04hr_train_df": sequences_04hr_train_df,
    "sequences_06hr_train_df": sequences_06hr_train_df,
    "sequences_08hr_train_df": sequences_08hr_train_df,
    "sequences_10hr_train_df": sequences_10hr_train_df,
    "sequences_12hr_train_df": sequences_12hr_train_df,
    "sequences_14hr_train_df": sequences_14hr_train_df,
    "sequences_16hr_train_df": sequences_16hr_train_df,
    "sequences_04hr_valid_np": sequences_04hr_valid_np,
    "sequences_06hr_valid_np": sequences_06hr_valid_np,
    "sequences_08hr_valid_np": sequences_08hr_valid_np,
    "sequences_10hr_valid_np": sequences_10hr_valid_np,
    "sequences_12hr_valid_np": sequences_12hr_valid_np,
    "sequences_14hr_valid_np": sequences_14hr_valid_np,
    "sequences_16hr_valid_np": sequences_16hr_valid_np,
    "sequences_04hr_valid_df": sequences_04hr_valid_df,
    "sequences_06hr_valid_df": sequences_06hr_valid_df,
    "sequences_08hr_valid_df": sequences_08hr_valid_df,
    "sequences_10hr_valid_df": sequences_10hr_valid_df,
    "sequences_12hr_valid_df": sequences_12hr_valid_df,
    "sequences_14hr_valid_df": sequences_14hr_valid_df,
    "sequences_16hr_valid_df": sequences_16hr_valid_df,
    "sequences_04hr_test_np": sequences_04hr_test_np,
    "sequences_06hr_test_np": sequences_06hr_test_np,
    "sequences_08hr_test_np": sequences_08hr_test_np,
    "sequences_10hr_test_np": sequences_10hr_test_np,
    "sequences_12hr_test_np": sequences_12hr_test_np,
    "sequences_14hr_test_np": sequences_14hr_test_np,
    "sequences_16hr_test_np": sequences_16hr_test_np,
    "sequences_04hr_test_df": sequences_04hr_test_df,
    "sequences_06hr_test_df": sequences_06hr_test_df,
    "sequences_08hr_test_df": sequences_08hr_test_df,
    "sequences_10hr_test_df": sequences_10hr_test_df,
    "sequences_12hr_test_df": sequences_12hr_test_df,
    "sequences_14hr_test_df": sequences_14hr_test_df,
    "sequences_16hr_test_df": sequences_16hr_test_df,
    "sequences_train_min": sequences_train_min,
    "sequences_train_max": sequences_train_max,
    "forecasts_train_df": forecasts_train_df,
    "forecasts_train_np": forecasts_train_np,
    "forecasts_valid_df": forecasts_valid_df,
    "forecasts_valid_np": forecasts_valid_np,
    "forecasts_test_df": forecasts_test_df,
    "forecasts_test_np": forecasts_test_np,
    "forecasts_train_min": forecasts_train_min,
    "forecasts_train_max": forecasts_train_max,
    "forecasts_first_dts_train": forecasts_first_dts_train,
    "forecasts_first_dts_valid": forecasts_first_dts_valid,
    "forecasts_first_dts_test": forecasts_first_dts_test
    }
pickle.dump(d_save, open("dataset_formatted.pkl", "wb"))