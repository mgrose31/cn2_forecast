# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 22:16:01 2021

@author: Mitchell Grose
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# %% formatted dataset
d = pickle.load(open("dataset_formatted.pkl", "rb"))
sequences_04hr_train = d.get("sequences_04hr_train_df")
sequences_08hr_train = d.get("sequences_08hr_train_df")
sequences_12hr_train = d.get("sequences_12hr_train_df")
sequences_16hr_train = d.get("sequences_16hr_train_df")
sequences_04hr_valid = d.get("sequences_04hr_valid_df")
sequences_08hr_valid = d.get("sequences_08hr_valid_df")
sequences_12hr_valid = d.get("sequences_12hr_valid_df")
sequences_16hr_valid = d.get("sequences_16hr_valid_df")
sequences_04hr_test = d.get("sequences_04hr_test_df")
sequences_08hr_test = d.get("sequences_08hr_test_df")
sequences_12hr_test = d.get("sequences_12hr_test_df")
sequences_16hr_test = d.get("sequences_16hr_test_df")
forecasts_train = d.get("forecasts_train_df")
forecasts_valid = d.get("forecasts_valid_df")
forecasts_test = d.get("forecasts_test_df")

# sequences
dts_sequences_train = np.empty((0, 32))
temp_train = np.empty((0, 32))
press_train = np.empty((0, 32))
rh_train = np.empty((0, 32))
wind_spd_train = np.empty((0, 32))
solar_irr_train = np.empty((0, 32))
cn2_sequences_train = np.empty((0, 32))
for x in sequences_16hr_train:
    dts_sequences_train = np.concatenate(
        (dts_sequences_train, np.transpose(
            np.expand_dims(x.index.to_pydatetime(), axis=-1))), axis=0)
    temp_train = np.concatenate(
        (temp_train, np.transpose(
            np.expand_dims(x['temp'].to_numpy(), axis=-1))), axis=0)
    press_train = np.concatenate(
        (press_train, np.transpose(
            np.expand_dims(x['press'].to_numpy(), axis=-1))), axis=0)
    rh_train = np.concatenate(
        (rh_train, np.transpose(
            np.expand_dims(x['rh'].to_numpy(), axis=-1))), axis=0)
    wind_spd_train = np.concatenate(
        (wind_spd_train, np.transpose(
            np.expand_dims(x['wind_spd'].to_numpy(), axis=-1))), axis=0)
    solar_irr_train = np.concatenate(
        (solar_irr_train, np.transpose(
            np.expand_dims(x['solar_irr'].to_numpy(), axis=-1))), axis=0)
    cn2_sequences_train = np.concatenate(
        (cn2_sequences_train, np.transpose(
            np.expand_dims(x['log10_cn2_night'].to_numpy(), axis=-1))), axis=0)

dts_sequences_valid = np.empty((0, 32))
temp_valid = np.empty((0, 32))
press_valid = np.empty((0, 32))
rh_valid = np.empty((0, 32))
wind_spd_valid = np.empty((0, 32))
solar_irr_valid = np.empty((0, 32))
cn2_sequences_valid = np.empty((0, 32))
for x in sequences_16hr_valid:
    dts_sequences_valid = np.concatenate(
        (dts_sequences_valid, np.transpose(
            np.expand_dims(x.index.to_pydatetime(), axis=-1))), axis=0)
    temp_valid = np.concatenate(
        (temp_valid, np.transpose(
            np.expand_dims(x['temp'].to_numpy(), axis=-1))), axis=0)
    press_valid = np.concatenate(
        (press_valid, np.transpose(
            np.expand_dims(x['press'].to_numpy(), axis=-1))), axis=0)
    rh_valid = np.concatenate(
        (rh_valid, np.transpose(
            np.expand_dims(x['rh'].to_numpy(), axis=-1))), axis=0)
    wind_spd_valid = np.concatenate(
        (wind_spd_valid, np.transpose(
            np.expand_dims(x['wind_spd'].to_numpy(), axis=-1))), axis=0)
    solar_irr_valid = np.concatenate(
        (solar_irr_valid, np.transpose(
            np.expand_dims(x['solar_irr'].to_numpy(), axis=-1))), axis=0)
    cn2_sequences_valid = np.concatenate(
        (cn2_sequences_valid, np.transpose(
            np.expand_dims(x['log10_cn2_night'].to_numpy(), axis=-1))), axis=0)

dts_sequences_test = np.empty((0, 32))
temp_test = np.empty((0, 32))
press_test = np.empty((0, 32))
rh_test = np.empty((0, 32))
wind_spd_test = np.empty((0, 32))
solar_irr_test = np.empty((0, 32))
cn2_sequences_test = np.empty((0, 32))
for x in sequences_16hr_test:
    dts_sequences_test = np.concatenate(
        (dts_sequences_test, np.transpose(
            np.expand_dims(x.index.to_pydatetime(), axis=-1))), axis=0)
    temp_test = np.concatenate(
        (temp_test, np.transpose(
            np.expand_dims(x['temp'].to_numpy(), axis=-1))), axis=0)
    press_test = np.concatenate(
        (press_test, np.transpose(
            np.expand_dims(x['press'].to_numpy(), axis=-1))), axis=0)
    rh_test = np.concatenate(
        (rh_test, np.transpose(
            np.expand_dims(x['rh'].to_numpy(), axis=-1))), axis=0)
    wind_spd_test = np.concatenate(
        (wind_spd_test, np.transpose(
            np.expand_dims(x['wind_spd'].to_numpy(), axis=-1))), axis=0)
    solar_irr_test = np.concatenate(
        (solar_irr_test, np.transpose(
            np.expand_dims(x['solar_irr'].to_numpy(), axis=-1))), axis=0)
    cn2_sequences_test = np.concatenate(
        (cn2_sequences_test, np.transpose(
            np.expand_dims(x['log10_cn2_night'].to_numpy(), axis=-1))), axis=0)


# forecasts
dts_forecasts_train = np.empty((0, 8))
ts_forecasts_train = np.empty((0, 8))
cn2_forecasts_train = np.empty((0, 8))
for x in forecasts_train:
    dts_forecasts_train = np.concatenate(
        (dts_forecasts_train, np.transpose(
            np.expand_dims(x.index.to_pydatetime(), axis=-1))), axis=0)
    ts_forecasts_train = np.concatenate(
        (ts_forecasts_train, np.transpose(
            np.expand_dims(x.index.time, axis=-1))), axis=0)
    cn2_forecasts_train = np.concatenate(
        (cn2_forecasts_train, np.transpose(
            np.expand_dims(x.to_numpy(), axis=-1))), axis=0)

dts_forecasts_valid = np.empty((0, 8))
ts_forecasts_valid = np.empty((0, 8))
cn2_forecasts_valid = np.empty((0, 8))
for x in forecasts_valid:
    dts_forecasts_valid = np.concatenate(
        (dts_forecasts_valid, np.transpose(
            np.expand_dims(x.index.to_pydatetime(), axis=-1))), axis=0)
    ts_forecasts_valid = np.concatenate(
        (ts_forecasts_valid, np.transpose(
            np.expand_dims(x.index.time, axis=-1))), axis=0)
    cn2_forecasts_valid = np.concatenate(
        (cn2_forecasts_valid, np.transpose(
            np.expand_dims(x.to_numpy(), axis=-1))), axis=0)
    
dts_forecasts_test = np.empty((0, 8))
ts_forecasts_test = np.empty((0, 8))
cn2_forecasts_test = np.empty((0, 8))
for x in forecasts_test:
    dts_forecasts_test = np.concatenate(
        (dts_forecasts_test, np.transpose(
            np.expand_dims(x.index.to_pydatetime(), axis=-1))), axis=0)
    ts_forecasts_test = np.concatenate(
        (ts_forecasts_test, np.transpose(
            np.expand_dims(x.index.time, axis=-1))), axis=0)
    cn2_forecasts_test = np.concatenate(
        (cn2_forecasts_test, np.transpose(
            np.expand_dims(x.to_numpy(), axis=-1))), axis=0)


first_dts_forecasts_train = pd.Series(d.get("forecasts_first_dts_train"))
first_dts_forecasts_valid = pd.Series(d.get("forecasts_first_dts_valid"))
first_dts_forecasts_test = pd.Series(d.get("forecasts_first_dts_test"))

first_ts_forecasts_train = first_dts_forecasts_train.dt.time
first_ts_forecasts_valid = first_dts_forecasts_valid.dt.time
first_ts_forecasts_test = first_dts_forecasts_test.dt.time

first_ts_forecasts_train_unique = np.unique(first_ts_forecasts_train)
first_ts_forecasts_valid_unique = np.unique(first_ts_forecasts_valid)
first_ts_forecasts_test_unique = np.unique(first_ts_forecasts_test)

idx0 = (first_ts_forecasts_train==first_ts_forecasts_valid_unique[-1]).to_numpy()

plt.figure()
for i, y in enumerate(cn2_forecasts_train[idx0]):
    if i%2==0: # use this to reduce number of curves
        plt.plot(10**y, '-o')
# plt.plot(10**cn2_forecasts_train[idx0], '-o')
plt.yscale('log')
plt.xlim(-34, 14)
plt.ylim(1e-17, 1e-14)
plt.ylabel("$C_{n}^{2}$ ($m^{-2/3}$)")
plt.grid(True, which='major')
plt.grid(True, which='minor')
plt.tight_layout()


