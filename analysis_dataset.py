# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 12:57:44 2021

@author: Mitchell Grose
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
# plt.style.use('ggplot')
plt.style.use('default')

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

dts_sequences_train_unique, idx_train_sequences_unique = np.unique(
    dts_sequences_train, return_index=True)
dts_sequences_valid_unique, idx_valid_sequences_unique = np.unique(
    dts_sequences_valid, return_index=True)
dts_sequences_test_unique, idx_test_sequences_unique = np.unique(
    dts_sequences_test, return_index=True)

temp_train_unique = temp_train.flatten()[idx_train_sequences_unique]
press_train_unique = press_train.flatten()[idx_train_sequences_unique]
rh_train_unique = rh_train.flatten()[idx_train_sequences_unique]
wind_spd_train_unique = wind_spd_train.flatten()[idx_train_sequences_unique]
solar_irr_train_unique = solar_irr_train.flatten()[idx_train_sequences_unique]
cn2_sequences_train_unique = cn2_sequences_train.flatten()[idx_train_sequences_unique]

temp_valid_unique = temp_valid.flatten()[idx_valid_sequences_unique]
press_valid_unique = press_valid.flatten()[idx_valid_sequences_unique]
rh_valid_unique = rh_valid.flatten()[idx_valid_sequences_unique]
wind_spd_valid_unique = wind_spd_valid.flatten()[idx_valid_sequences_unique]
solar_irr_valid_unique = solar_irr_valid.flatten()[idx_valid_sequences_unique]
cn2_sequences_valid_unique = cn2_sequences_valid.flatten()[idx_valid_sequences_unique]

temp_test_unique = temp_test.flatten()[idx_test_sequences_unique]
press_test_unique = press_test.flatten()[idx_test_sequences_unique]
rh_test_unique = rh_test.flatten()[idx_test_sequences_unique]
wind_spd_test_unique = wind_spd_test.flatten()[idx_test_sequences_unique]
solar_irr_test_unique = solar_irr_test.flatten()[idx_test_sequences_unique]
cn2_sequences_test_unique = cn2_sequences_test.flatten()[idx_test_sequences_unique]

# forecasts
dts_forecasts_train = np.empty((0, 8))
cn2_forecasts_train = np.empty((0, 8))
for x in forecasts_train:
    dts_forecasts_train = np.concatenate(
        (dts_forecasts_train, np.transpose(
            np.expand_dims(x.index.to_pydatetime(), axis=-1))), axis=0)
    cn2_forecasts_train = np.concatenate(
        (cn2_forecasts_train, np.transpose(
            np.expand_dims(x.to_numpy(), axis=-1))), axis=0)

dts_forecasts_valid = np.empty((0, 8))
cn2_forecasts_valid = np.empty((0, 8))
for x in forecasts_valid:
    dts_forecasts_valid = np.concatenate(
        (dts_forecasts_valid, np.transpose(
            np.expand_dims(x.index.to_pydatetime(), axis=-1))), axis=0)
    cn2_forecasts_valid = np.concatenate(
        (cn2_forecasts_valid, np.transpose(
            np.expand_dims(x.to_numpy(), axis=-1))), axis=0)
    
dts_forecasts_test = np.empty((0, 8))
cn2_forecasts_test = np.empty((0, 8))
for x in forecasts_test:
    dts_forecasts_test = np.concatenate(
        (dts_forecasts_test, np.transpose(
            np.expand_dims(x.index.to_pydatetime(), axis=-1))), axis=0)
    cn2_forecasts_test = np.concatenate(
        (cn2_forecasts_test, np.transpose(
            np.expand_dims(x.to_numpy(), axis=-1))), axis=0)

dts_forecasts_train_unique, idx_train_forecasts_unique = np.unique(
    dts_forecasts_train, return_index=True)
dts_forecasts_valid_unique, idx_valid_forecasts_unique = np.unique(
    dts_forecasts_valid, return_index=True)
dts_forecasts_test_unique, idx_test_forecasts_unique = np.unique(
    dts_forecasts_test, return_index=True)

cn2_forecasts_train_unique = cn2_forecasts_train.flatten()[idx_train_forecasts_unique]
cn2_forecasts_valid_unique = cn2_forecasts_valid.flatten()[idx_valid_forecasts_unique]
cn2_forecasts_test_unique = cn2_forecasts_test.flatten()[idx_test_forecasts_unique]

# temporal plots
plt.figure(figsize=(8, 5))
plt.plot(dts_sequences_train_unique, temp_train_unique, 'k.', label='train')
plt.plot(dts_sequences_valid_unique, temp_valid_unique, 'b.', label='valid')
plt.plot(dts_sequences_test_unique, temp_test_unique, 'r.', label='test')
plt.xlabel('local time (EST)')
plt.ylabel('temperature (K)')
# plt.xticks(rotation=30)
plt.legend(loc='lower right')
plt.grid(True, which='major')
plt.grid(True, which='minor')
plt.tight_layout()

plt.figure(figsize=(8, 5))
plt.plot(dts_sequences_train_unique, press_train_unique, 'k.', label='train')
plt.plot(dts_sequences_valid_unique, press_valid_unique, 'b.', label='valid')
plt.plot(dts_sequences_test_unique, press_test_unique, 'r.', label='test')
plt.xlabel('local time (EST)')
plt.ylabel('pressure (Pa)')
# plt.xticks(rotation=30)
plt.legend(loc='lower right')
plt.grid(True, which='major')
plt.grid(True, which='minor')
plt.tight_layout()

plt.figure(figsize=(8, 5))
plt.plot(dts_sequences_train_unique, rh_train_unique, 'k.', label='train')
plt.plot(dts_sequences_valid_unique, rh_valid_unique, 'b.', label='valid')
plt.plot(dts_sequences_test_unique, rh_test_unique, 'r.', label='test')
plt.ylim(0, 100)
plt.xlabel('local time (EST)')
plt.ylabel('relative humidity (%)')
# plt.xticks(rotation=30)
plt.legend(loc='lower right')
plt.grid(True, which='major')
plt.grid(True, which='minor')
plt.tight_layout()

plt.figure(figsize=(8, 5))
plt.plot(dts_sequences_train_unique, wind_spd_train_unique, 'k.', label='train')
plt.plot(dts_sequences_valid_unique, wind_spd_valid_unique, 'b.', label='valid')
plt.plot(dts_sequences_test_unique, wind_spd_test_unique, 'r.', label='test')
plt.xlabel('local time (EST)')
plt.ylabel('wind speed (m/s)')
# plt.xticks(rotation=30)
plt.legend(loc='upper right')
plt.grid(True, which='major')
plt.grid(True, which='minor')
plt.tight_layout()

plt.figure(figsize=(8, 5))
plt.plot(dts_sequences_train_unique, solar_irr_train_unique, 'k.', label='train')
plt.plot(dts_sequences_valid_unique, solar_irr_valid_unique, 'b.', label='valid')
plt.plot(dts_sequences_test_unique, solar_irr_test_unique, 'r.', label='test')
plt.ylim(0, 1200)
plt.xlabel('local time (EST)')
plt.ylabel('solar irradiance ($W/m^{2}$)')
# plt.xticks(rotation=30)
plt.legend(loc='upper right')
plt.grid(True, which='major')
plt.grid(True, which='minor')
plt.tight_layout()

plt.figure(figsize=(8, 5))
plt.plot(dts_sequences_train_unique, 10**cn2_sequences_train_unique, 'k.', label='train')
plt.plot(dts_sequences_valid_unique, 10**cn2_sequences_valid_unique, 'b.', label='valid')
plt.plot(dts_sequences_test_unique, 10**cn2_sequences_test_unique, 'r.', label='test')
plt.yscale('log')
plt.ylim(1e-17, 1e-14)
plt.xlabel('local time (EST)')
plt.ylabel('sequence $C_{n}^{2}$ ($m^{-2/3}$)')
# plt.xticks(rotation=30)
plt.legend(loc='lower right')
plt.grid(True, which='major')
plt.grid(True, which='minor')
plt.tight_layout()

plt.figure(figsize=(8, 5))
plt.plot(dts_forecasts_train_unique, 10**cn2_forecasts_train_unique, 'k.', label='train')
plt.plot(dts_forecasts_valid_unique, 10**cn2_forecasts_valid_unique, 'b.', label='valid')
plt.plot(dts_forecasts_test_unique, 10**cn2_forecasts_test_unique, 'r.', label='test')
plt.yscale('log')
plt.ylim(1e-17, 1e-14)
plt.xlabel('local time (EST)')
plt.ylabel('forecast $C_{n}^{2}$ ($m^{-2/3}$)')
# plt.xticks(rotation=30)
plt.legend(loc='lower right')
plt.grid(True, which='major')
plt.grid(True, which='minor')
plt.tight_layout()

# histogram plots
nbins = 75
_, bins = np.histogram(
    np.concatenate(
        (temp_train_unique, temp_valid_unique, temp_test_unique)), bins=nbins)
plt.figure()
plt.hist(temp_train_unique, bins=bins, density=True, alpha=1, color='k', label='train')
plt.hist(temp_valid_unique, bins=bins, density=True, alpha=0.7, color='b', label='valid')
plt.hist(temp_test_unique, bins=bins, density=True, alpha=0.5, color='r', label='test')
plt.xlabel('temperature (K)')
plt.ylabel('probability density')
plt.legend(loc='best')
plt.grid(True, which='major')
plt.grid(True, which='minor')
plt.tight_layout()


_, bins = np.histogram(
    np.concatenate(
        (press_train_unique, press_valid_unique, press_test_unique)), bins=nbins)
plt.figure()
plt.hist(press_train_unique, bins=bins, density=True, alpha=1, color='k', label='train')
plt.hist(press_valid_unique, bins=bins, density=True, alpha=0.7, color='b', label='valid')
plt.hist(press_test_unique, bins=bins, density=True, alpha=0.5, color='r', label='test')
plt.xlabel('pressure (Pa)')
plt.ylabel('probability density')
plt.legend(loc='best')
plt.grid(True, which='major')
plt.grid(True, which='minor')
plt.tight_layout()

_, bins = np.histogram(
    np.concatenate(
        (rh_train_unique, rh_valid_unique, rh_test_unique)), bins=nbins)
plt.figure()
plt.hist(rh_train_unique, bins=bins, density=True, alpha=1, color='k', label='train')
plt.hist(rh_valid_unique, bins=bins, density=True, alpha=0.7, color='b', label='valid')
plt.hist(rh_test_unique, bins=bins, density=True, alpha=0.5, color='r', label='test')
plt.xlim(0, 100)
plt.xlabel('relative humidity (%)')
plt.ylabel('probability density')
plt.legend(loc='best')
plt.grid(True, which='major')
plt.grid(True, which='minor')
plt.tight_layout()

_, bins = np.histogram(
    np.concatenate(
        (wind_spd_train_unique, wind_spd_valid_unique, wind_spd_test_unique)), bins=nbins)
plt.figure()
plt.hist(wind_spd_train_unique, bins=bins, density=True,
         log=True, alpha=1, color='k', label='train')
plt.hist(wind_spd_valid_unique, bins=bins, density=True,
         log=True, alpha=0.7, color='b', label='valid')
plt.hist(wind_spd_test_unique, bins=bins, density=True,
         log=True, alpha=0.5, color='r', label='test')
plt.xlabel('wind speed (m/s)')
plt.ylabel('probability density')
plt.legend(loc='best')
plt.grid(True, which='major')
plt.grid(True, which='minor')
plt.tight_layout()

_, bins = np.histogram(
    np.concatenate(
        (solar_irr_train_unique, solar_irr_valid_unique, solar_irr_test_unique)), bins=nbins)
plt.figure()
plt.hist(solar_irr_train_unique, bins=bins, density=True,
         log=True, alpha=1, color='k', label='train')
plt.hist(solar_irr_valid_unique, bins=bins, density=True,
         log=True, alpha=0.7, color='b', label='valid')
plt.hist(solar_irr_test_unique, bins=bins, density=True,
         log=True, alpha=0.5, color='r', label='test')
plt.xlabel('solar irradiance ($W/m^{2}$)')
plt.ylabel('probability density')
plt.legend(loc='best')
plt.grid(True, which='major')
plt.grid(True, which='minor')
plt.tight_layout()

_, bins = np.histogram(
    np.concatenate(
        (cn2_sequences_train_unique, cn2_sequences_valid_unique, cn2_sequences_test_unique)), bins=nbins)
plt.figure()
plt.hist(cn2_sequences_train_unique, bins=bins, density=True,
         log=True, alpha=1, color='k', label='train')
plt.hist(cn2_sequences_valid_unique, bins=bins, density=True,
         log=True, alpha=0.7, color='b', label='valid')
plt.hist(cn2_sequences_test_unique, bins=bins, density=True,
         log=True, alpha=0.5, color='r', label='test')
plt.xlim(-17, -14)
plt.gca().set_ylim(bottom=0.01, top=20)
plt.xlabel('sequence $log_{10}(C_{n}^{2})$')
plt.ylabel('probability density')
plt.legend(loc='best')
plt.grid(True, which='major')
plt.grid(True, which='minor')
plt.tight_layout()

_, bins = np.histogram(
    np.concatenate(
        (cn2_forecasts_train_unique, cn2_forecasts_valid_unique, cn2_forecasts_test_unique)), bins=nbins)
plt.figure()
plt.hist(cn2_forecasts_train_unique, bins=bins, density=True,
         log=True, alpha=1, color='k', label='train')
plt.hist(cn2_forecasts_valid_unique, bins=bins, density=True,
         log=True, alpha=0.7, color='b', label='valid')
plt.hist(cn2_forecasts_test_unique, bins=bins, density=True,
         log=True, alpha=0.5, color='r', label='test')
plt.xlim(-17, -14)
plt.gca().set_ylim(bottom=0.01, top=20)
plt.xlabel('forecast $log_{10}(C_{n}^{2})$')
plt.ylabel('probability density')
plt.legend(loc='best')
plt.grid(True, which='major')
plt.grid(True, which='minor')
plt.tight_layout()

###############################################################################
# %% unformatted dataset
# dataset = pickle.load(open('dataset_unformatted.pkl', 'rb'))

# log10_cn2_df = dataset['log10_cn2']
# log10_cn2_np = log10_cn2_df.to_numpy().copy()

# # dts = dataset.index.to_pydatetime()
# dts = dataset.index
# idx_night = np.logical_or(dts.hour.to_numpy() <= 7,
#                           dts.hour.to_numpy() >= 21)
# idx_night = np.logical_and(idx_night,
#                            np.isnan(log10_cn2_np))
# log10_cn2_tmp = log10_cn2_np.copy()
# log10_cn2_tmp[idx_night] = -15
# dataset['log10_cn2_night'] = log10_cn2_tmp

# train_dt_start = datetime(2020, 6, 1, 0, 0, 0)
# train_dt_end = datetime(2020, 7, 18, 0, 0, 0)
# valid_dt_start = train_dt_end
# valid_dt_end = datetime(2020, 8, 2, 0, 0, 0)
# test_dt_start = valid_dt_end
# test_dt_end = datetime(2020, 8, 11, 0, 0, 0)

# idx_train = np.logical_and(dts >= train_dt_start,
#                            dts < train_dt_end)
# idx_valid = np.logical_and(dts >= valid_dt_start,
#                            dts < valid_dt_end)
# idx_test = np.logical_and(dts >= test_dt_start,
#                           dts < test_dt_end)

# plt.figure(figsize=(8,5))
# plt.plot(dts[idx_train], dataset['temp'].loc[idx_train], 'k.', label='train')
# plt.plot(dts[idx_valid], dataset['temp'].loc[idx_valid], 'b.', label='valid')
# plt.plot(dts[idx_test], dataset['temp'].loc[idx_test], 'r.', label='test')
# plt.xlabel('local time (EST)')
# plt.ylabel('temperature (K)')
# plt.xticks(rotation=30)
# plt.legend(loc='best')
# plt.grid(True, which='major')
# plt.grid(True, which='minor')
# plt.tight_layout()

# plt.figure(figsize=(8,5))
# plt.plot(dts[idx_train], dataset['press'].loc[idx_train], 'k.', label='train')
# plt.plot(dts[idx_valid], dataset['press'].loc[idx_valid], 'b.', label='valid')
# plt.plot(dts[idx_test], dataset['press'].loc[idx_test], 'r.', label='test')
# plt.xlabel('local time (EST)')
# plt.ylabel('pressure (Pa)')
# plt.xticks(rotation=30)
# plt.legend(loc='best')
# plt.grid(True, which='major')
# plt.grid(True, which='minor')
# plt.tight_layout()

# plt.figure(figsize=(8,5))
# plt.plot(dts[idx_train], dataset['rh'].loc[idx_train], 'k.', label='train')
# plt.plot(dts[idx_valid], dataset['rh'].loc[idx_valid], 'b.', label='valid')
# plt.plot(dts[idx_test], dataset['rh'].loc[idx_test], 'r.', label='test')
# plt.ylim(0, 100)
# plt.xlabel('local time (EST)')
# plt.ylabel('relative humidity (%)')
# plt.xticks(rotation=30)
# plt.legend(loc='best')
# plt.grid(True, which='major')
# plt.grid(True, which='minor')
# plt.tight_layout()

# plt.figure(figsize=(8,5))
# plt.plot(dts[idx_train], dataset['wind_spd'].loc[idx_train], 'k.', label='train')
# plt.plot(dts[idx_valid], dataset['wind_spd'].loc[idx_valid], 'b.', label='valid')
# plt.plot(dts[idx_test], dataset['wind_spd'].loc[idx_test], 'r.', label='test')
# plt.xlabel('local time (EST)')
# plt.ylabel('wind speed (m/s)')
# plt.xticks(rotation=30)
# plt.legend(loc='best')
# plt.grid(True, which='major')
# plt.grid(True, which='minor')
# plt.tight_layout()

# plt.figure(figsize=(8,5))
# plt.plot(dts[idx_train], dataset['solar_irr'].loc[idx_train], 'k.', label='train')
# plt.plot(dts[idx_valid], dataset['solar_irr'].loc[idx_valid], 'b.', label='valid')
# plt.plot(dts[idx_test], dataset['solar_irr'].loc[idx_test], 'r.', label='test')
# plt.xlabel('local time (EST)')
# plt.ylabel('solar irradiance ($W/m^{2}$)')
# plt.xticks(rotation=30)
# plt.legend(loc='upper left')
# plt.grid(True, which='major')
# plt.grid(True, which='minor')
# plt.tight_layout()

# plt.figure(figsize=(8,5))
# plt.plot(dts[idx_train], 10**dataset['log10_cn2_night'].loc[idx_train], 'k.', label='train')
# plt.plot(dts[idx_valid], 10**dataset['log10_cn2_night'].loc[idx_valid], 'b.', label='valid')
# plt.plot(dts[idx_test], 10**dataset['log10_cn2_night'].loc[idx_test], 'r.', label='test')
# plt.yscale('log')
# plt.ylim(1e-17, 1e-14)
# plt.xlabel('local time (EST)')
# plt.ylabel('$C_{n}^{2}$ ($m^{-2/3}$)')
# plt.xticks(rotation=30)
# plt.legend(loc='best')
# plt.grid(True, which='major')
# plt.grid(True, which='minor')
# plt.tight_layout()