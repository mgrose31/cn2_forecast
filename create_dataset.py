# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 17:27:10 2020

@author: mgrose
"""
# %% import libraries
import os
import pandas as pd
import numpy as np
from helpers import read_davis_weather, extract_DELTA_summary_data_single_dir
from helpers import window_average
import matplotlib.pyplot as plt
from datetime import date, datetime, timedelta

win_hms = [0,30,0]
int_hms = [0,1,0]

# %% read in the weather and turbulence data
path_wx = os.path.abspath(r"D:\Documents\EOP Program\Research\cn2_forecast\data\MZA_Wx_data\2020_WxData.txt")
path_cn2 = os.path.abspath(r"D:\Documents\EOP Program\Research\cn2_forecast\data\Fitz_Hall_DELTA_data")
df_wx = read_davis_weather(os.path.abspath(path_wx))
df_cn2 = extract_DELTA_summary_data_single_dir(os.path.abspath(path_cn2), 50, 70)

# %% convert air density and pressure to correct values at certain times
air_density = df_wx['air_density']
air_density_indices = air_density <= 0.5 # Get indices where air density is in lb/ft^3
df_wx.loc[air_density_indices, 'air_density'] = \
    df_wx.loc[air_density_indices, 'air_density'] * 16.02 # Convert lb/ft^3 to kg/m^3
    
press_indices = df_wx.loc[:, 'DATE'].dt.to_pydatetime() <= datetime(2020,4,21,17,25,0)
df_wx.loc[press_indices, 'press'] = round(df_wx.loc[press_indices, 'press'] * (0.9690), -1)

# %% plot turbulence (cn2) and weather data
# plt.figure(figsize=(10, 6))
# plt.plot(df_cn2['DATE'], df_cn2['cn2'], 'k.', markersize=4)
# plt.yscale('log')
# plt.ylim(1e-17, 1e-13)
# plt.xlabel('local time (EST)')
# plt.ylabel('$C_{n}^{2} (m^{-2/3})$')
# plt.xticks(rotation=30)
# plt.grid(True)
# plt.tight_layout()

# plt.figure(figsize=(10, 6))
# plt.plot(df_wx['DATE'], df_wx['temp'], 'k.', markersize=4)
# plt.xlabel('local time (EST)')
# plt.ylabel('temperature (K)')
# plt.xticks(rotation=30)
# plt.grid(True)
# plt.tight_layout()

# plt.figure(figsize=(10, 6))
# plt.plot(df_wx['DATE'], df_wx['press'], 'c.', markersize=4)
# plt.xlabel('local time (EST)')
# plt.ylabel('pressure (Pa)')
# plt.xticks(rotation=30)
# plt.grid(True)
# plt.tight_layout()

# plt.figure(figsize=(10, 6))
# plt.plot(df_wx['DATE'], df_wx['rh'], 'm.', markersize=4)
# plt.ylim(0, 100)
# plt.xlabel('local time (EST)')
# plt.ylabel('relative humidity (%)')
# plt.xticks(rotation=30)
# plt.grid(True)
# plt.tight_layout()

# plt.figure(figsize=(10, 6))
# plt.plot(df_wx['DATE'], df_wx['rain_rate'], 'y.', markersize=4)
# plt.xlabel('local time (EST)')
# plt.ylabel('rain rate')
# plt.xticks(rotation=30)
# plt.grid(True)
# plt.tight_layout()

# plt.figure(figsize=(10, 6))
# plt.plot(df_wx['DATE'], df_wx['u_wind'], 'g.', markersize=4)
# plt.ylim(-10, 10)
# plt.xlabel('local time (EST)')
# plt.ylabel('u_wind (m/s)')
# plt.xticks(rotation=30)
# plt.grid(True)
# plt.tight_layout()

# plt.figure(figsize=(10, 6))
# plt.plot(df_wx['DATE'], df_wx['v_wind'], 'b.', markersize=4)
# plt.ylim(-10, 10)
# plt.xlabel('local time (EST)')
# plt.ylabel('v_wind (m/s)')
# plt.xticks(rotation=30)
# plt.grid(True)
# plt.tight_layout()

# plt.figure(figsize=(10, 6))
# plt.plot(df_wx['DATE'], df_wx['solar_irr'], 'r.', markersize=4)
# plt.xlabel('local time (EST)')
# plt.ylabel('solar irradiance ($W/m^{2}$)')
# plt.xticks(rotation=30)
# plt.grid(True)
# plt.tight_layout()

# %% format weather data
start_day = date(2020, 4, 12)
end_day = date(2020, 8, 11)

wx_date_range = pd.date_range(start_day, end_day, freq='30min')
wx_dates = df_wx['DATE']

print("Window averaging temperature...")
df_win_avg = window_average(x=df_wx['temp'],
                            t=wx_dates,
                            win_hms=win_hms, int_hms=int_hms)
t = df_win_avg['t_win_avg']
temp = df_win_avg['x_win_avg'].to_numpy()

print("Window averaging pressure...")
df_win_avg = window_average(x=df_wx['press'],
                            t=wx_dates,
                            win_hms=win_hms, int_hms=int_hms)
press = df_win_avg['x_win_avg'].to_numpy()

print("Window averaging relative humidity...")
df_win_avg = window_average(x=df_wx['rh'],
                            t=wx_dates,
                            win_hms=win_hms, int_hms=int_hms)
rh = df_win_avg['x_win_avg'].to_numpy()

print("Window averaging wind speed...")
df_win_avg = window_average(x=df_wx['wind_speed'],
                            t=wx_dates,
                            win_hms=win_hms, int_hms=int_hms)
wind_speed = df_win_avg['x_win_avg'].to_numpy()

print("Window averaging solar irradiance...")
df_win_avg = window_average(x=df_wx['solar_irr'],
                            t=wx_dates,
                            win_hms=win_hms, int_hms=int_hms)
solar_irr = df_win_avg['x_win_avg'].to_numpy()

# plt.figure()
# plt.plot(t, temp, 'k.')
# plt.grid(True)
# plt.grid(True, 'minor')
# plt.tight_layout()

# plt.figure()
# plt.plot(t, press, 'c.')
# plt.grid(True)
# plt.grid(True, 'minor')
# plt.tight_layout()

# plt.figure()
# plt.plot(t, rh, 'm.')
# plt.ylim(0, 100)
# plt.grid(True)
# plt.grid(True, 'minor')
# plt.tight_layout()

# plt.figure()
# plt.plot(t, wind_speed, 'b.')
# plt.grid(True)
# plt.grid(True, 'minor')
# plt.tight_layout()

# plt.figure()
# plt.plot(t, solar_irr, 'r.')
# plt.grid(True)
# plt.grid(True, 'minor')
# plt.tight_layout()

temp_interp = np.interp(wx_date_range, t, temp, np.nan, np.nan)
press_interp = np.interp(wx_date_range, t, press, np.nan, np.nan)
rh_interp = np.interp(wx_date_range, t, rh, np.nan, np.nan)
windspd_interp = np.interp(wx_date_range, t, wind_speed, np.nan, np.nan)
solarirr_interp = np.interp(wx_date_range, t, solar_irr, np.nan, np.nan)


dict_wx_sample = {
    'DATE': wx_date_range,
    'temp': temp_interp,
    'press': press_interp,
    'rh': rh_interp,
    'wind_spd': windspd_interp,
    'solar_irr': solarirr_interp}
df_wx_sample = pd.DataFrame(dict_wx_sample)

# get indices of interplations greater than 30 minutes
wx_idx = np.array([])
for this_time in wx_date_range:
    wx_idx = np.append(wx_idx, any(abs(wx_dates-this_time)<timedelta(minutes=30)))
print("Removing {} interpolated weather measurements...".format(int(len(wx_idx) - wx_idx.sum())))
df_wx_sample.loc[wx_idx==0] = np.nan
df_wx_sample['DATE'] = wx_date_range

# plot interpolated weather data
plt.figure(figsize=(10, 6))
plt.plot(df_wx_sample['DATE'], df_wx_sample['temp'], 'k.', markersize=4)
plt.title("Sampled Weather: Temperature")
plt.ylabel("Temperature (K)")
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()

plt.figure(figsize=(10, 6))
plt.plot(df_wx_sample['DATE'], df_wx_sample['press'], 'c.', markersize=4)
plt.title("Sampled Weather: Pressure")
plt.ylabel("Pressure (Pa)")
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()

plt.figure(figsize=(10, 6))
plt.plot(df_wx_sample['DATE'], df_wx_sample['rh'], 'm.', markersize=4)
plt.ylim(0, 100)
plt.title("Sampled Weather: Relative Humidity")
plt.ylabel("Relative Humidity (%)")
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()

plt.figure(figsize=(10, 6))
plt.plot(df_wx_sample['DATE'], df_wx_sample['wind_spd'], 'y.', markersize=4)
plt.title("Sampled Weather: wind speed")
plt.ylabel("wind speed (m/s)")
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()

plt.figure(figsize=(10, 6))
plt.plot(df_wx_sample['DATE'], df_wx_sample['solar_irr'], 'r.', markersize=4)
plt.title("Sampled Weather: Solar Irradiance")
plt.ylabel("Solar Irradiance ($W/m^{2}$)")
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()

# %% format turbulence dataset
print("Window averaging...")
df_cn2_win_avg = window_average(x=df_cn2['log10_cn2'],
                                t=df_cn2['DATE'],
                                win_hms=win_hms, int_hms=int_hms)

dates_cn2 = pd.to_datetime(df_cn2_win_avg['t_win_avg'])

log10_cn2_interp = np.interp(wx_date_range, dates_cn2, df_cn2_win_avg['x_win_avg'], np.nan, np.nan)
dict_log10_cn2_sample = {
    'DATE': wx_date_range,
    'log10_cn2': log10_cn2_interp}
df_cn2_sample = pd.DataFrame(dict_log10_cn2_sample)


idx1 = np.array([]);
for this_time in wx_date_range:
    idx1 = np.append(idx1, any(abs(dates_cn2-this_time)<timedelta(minutes=30)))
print("Removing {} interplated turbulence measurements...".format(int(len(idx1) - idx1.sum())))

log10_cn2_tmp = df_cn2_sample['log10_cn2'].to_numpy()
log10_cn2_tmp[idx1==0] = np.nan
df_cn2_sample['log10_cn2'] = log10_cn2_tmp
# df_cn2_sample['log10_cn2'].iloc[idx1==0] = np.nan

# %% combine weather and turbulence datasets into one
df_wx_sample.index = df_wx_sample['DATE']
df_wx_sample = df_wx_sample.drop(columns='DATE')
df_cn2_sample.index = df_cn2_sample['DATE']
df_cn2_sample = df_cn2_sample.drop(columns='DATE')

dataset = pd.concat([df_wx_sample, df_cn2_sample], axis=1)

# # %%
# plt.figure(figsize=(10, 6))
# plt.plot(dataset.index, dataset['temp'], 'k.')
# plt.grid(True)
# plt.grid(True, 'minor')
# plt.tight_layout()

# plt.figure(figsize=(10, 6))
# plt.plot(dataset.index, dataset['press'], 'k.')
# plt.grid(True)
# plt.grid(True, 'minor')
# plt.tight_layout()

# plt.figure(figsize=(10, 6))
# plt.plot(dataset.index, dataset['rh'], 'k.')
# plt.grid(True)
# plt.grid(True, 'minor')
# plt.tight_layout()

# plt.figure(figsize=(10, 6))
# plt.plot(dataset.index, dataset['wind_spd'], 'k.')
# plt.grid(True)
# plt.grid(True, 'minor')
# plt.tight_layout()

# plt.figure(figsize=(10, 6))
# plt.plot(dataset.index, dataset['solar_irr'], 'k.')
# plt.grid(True)
# plt.grid(True, 'minor')
# plt.tight_layout()

# %%
# dataset.to_pickle('dataset_30min.pkl')