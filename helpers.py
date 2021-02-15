# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 17:30:01 2020

@author: mgrose
"""
import os
import h5py
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def read_davis_weather(filename):        
    # Read file by lines, then split each line into fields
    wxFile = open(filename, 'r')
    testStr = wxFile.readlines()
    
    if testStr[2][0]=="-":
        file_type_flag = 1
        start_idx = 3
    else:
        file_type_flag = 0
        start_idx = 2
    
    for ii in range(0, len(testStr)):
        if file_type_flag!=1:
            testStr[ii] = testStr[ii].split("\t") # Use this for manually exported .txt files since they're tab delimited
        elif file_type_flag==1:
            # Use these for automatically exported .txt file, 'download.txt', since it's delimited by multiple combinations of spaces
            testStr[ii] = testStr[ii].lstrip() # Remove whitespace at left side of string
            testStr[ii] = testStr[ii].rstrip() # Remove whitespace at right side of string
            testStr[ii] = testStr[ii].replace("      ", ",") # Six spaces
            testStr[ii] = testStr[ii].replace("     ", ",") # Five spaces
            testStr[ii] = testStr[ii].replace("    ", ",") # Four spaces
            testStr[ii] = testStr[ii].replace("   ", ",") # Three spaces
            testStr[ii] = testStr[ii].replace("  ", ",") # Two spaces
            testStr[ii] = testStr[ii].replace(" ", ",") # One space
            testStr[ii] = testStr[ii].split(",") # Split by field
    
    wxFile.close()
    
    weather_df = pd.DataFrame(testStr).drop(list(range(start_idx)), axis=0)
    

    mm_dd_yyyy = weather_df.iloc[:, 0].reset_index(drop=True)
    time_ampm = weather_df.iloc[:, 1].reset_index(drop=True)
    temp_out = pd.to_numeric(
        weather_df.iloc[:, 2].reset_index(drop=True), errors='coerce')
    temp_hi = pd.to_numeric(
        weather_df.iloc[:, 3].reset_index(drop=True), errors='coerce')
    temp_low = pd.to_numeric(
        weather_df.iloc[:, 4].reset_index(drop=True), errors='coerce')
    hum_out = pd.to_numeric(
        weather_df.iloc[:, 5].reset_index(drop=True), errors='coerce')
    dew_pt = pd.to_numeric(
        weather_df.iloc[:, 6].reset_index(drop=True), errors='coerce')
    wind_speed = pd.to_numeric(
        weather_df.iloc[:, 7].reset_index(drop=True), errors='coerce')
    wind_dir = weather_df.iloc[:, 8].reset_index(drop=True)
    wind_run = pd.to_numeric(
        weather_df.iloc[:, 9].reset_index(drop=True), errors='coerce')
    wind_speed_hi = pd.to_numeric(
        weather_df.iloc[:, 10].reset_index(drop=True), errors='coerce')
    wind_dir_hi = weather_df.iloc[:, 11].reset_index(drop=True)
    wind_chill = pd.to_numeric(
        weather_df.iloc[:, 12].reset_index(drop=True), errors='coerce')
    head_index = pd.to_numeric(
        weather_df.iloc[:, 13].reset_index(drop=True), errors='coerce')
    THW_index = pd.to_numeric(
        weather_df.iloc[:, 14].reset_index(drop=True), errors='coerce')
    THSW_index = pd.to_numeric(
        weather_df.iloc[:, 15].reset_index(drop=True), errors='coerce')
    press = pd.to_numeric(
        weather_df.iloc[:, 16].reset_index(drop=True), errors='coerce')
    rain = pd.to_numeric(
        weather_df.iloc[:, 17].reset_index(drop=True), errors='coerce')
    rain_rate = pd.to_numeric(
        weather_df.iloc[:, 18].reset_index(drop=True), errors='coerce')
    solar_irr = pd.to_numeric(
        weather_df.iloc[:, 19].reset_index(drop=True), errors='coerce')
    solar_energy = pd.to_numeric(
        weather_df.iloc[:, 20].reset_index(drop=True), errors='coerce')
    solar_irr_hi = pd.to_numeric(
        weather_df.iloc[:, 21].reset_index(drop=True), errors='coerce')
    UV_index = pd.to_numeric(
        weather_df.iloc[:, 22].reset_index(drop=True), errors='coerce')
    UV_dose = pd.to_numeric(
        weather_df.iloc[:, 23].reset_index(drop=True), errors='coerce')
    UV_Hi = pd.to_numeric(
        weather_df.iloc[:, 24].reset_index(drop=True), errors='coerce')
    heat_DD = pd.to_numeric(
        weather_df.iloc[:, 25].reset_index(drop=True), errors='coerce')
    cool_DD = pd.to_numeric(
        weather_df.iloc[:, 26].reset_index(drop=True), errors='coerce')
    temp_in = pd.to_numeric(
        weather_df.iloc[:, 27].reset_index(drop=True), errors='coerce')
    hum_in = pd.to_numeric(
        weather_df.iloc[:, 28].reset_index(drop=True), errors='coerce')
    dew_pt_in = pd.to_numeric(
        weather_df.iloc[:, 29].reset_index(drop=True), errors='coerce')
    heat_in = pd.to_numeric(
        weather_df.iloc[:, 30].reset_index(drop=True), errors='coerce')
    EMC_in = pd.to_numeric(
        weather_df.iloc[:, 31].reset_index(drop=True), errors='coerce')
    air_density_in = pd.to_numeric(
        weather_df.iloc[:, 32].reset_index(drop=True), errors='coerce')
    ET = pd.to_numeric(
        weather_df.iloc[:, 33].reset_index(drop=True), errors='coerce')
    wind_samp = pd.to_numeric(
        weather_df.iloc[:, 34].reset_index(drop=True), errors='coerce')
    wind_tx = pd.to_numeric(
        weather_df.iloc[:, 35].reset_index(drop=True), errors='coerce')
    ISS_recept = pd.to_numeric(
        weather_df.iloc[:, 36].reset_index(drop=True), errors='coerce')
    arc_int = pd.to_numeric(
        weather_df.iloc[:, 37].reset_index(drop=True), errors='coerce')

    date_time = pd.to_datetime(mm_dd_yyyy + " " + time_ampm)

    time_zone = np.full(len(date_time), 'America/New_York')
    
    # Convert Units
    temp_out = (temp_out - 32) * (5/9) + 273.15   # Convert to K
    temp_hi = (temp_hi - 32) * (5/9) + 273.15   # Convert to K
    temp_low = (temp_low - 32) * (5/9) + 273.15 # Convert to K

    dew_pt = (dew_pt - 32) * (5/9) + 273.15

    press = press*100 # Convert to Pa

    # Define function to assign wind compass nominal directions to degrees
    def windCompass2Deg(wind_compass_dirs):
        wind_dir_deg = np.empty(len(wind_compass_dirs))

        N   = wind_compass_dirs == 'N'
        NNE = wind_compass_dirs == 'NNE'
        NE  = wind_compass_dirs == 'NE'
        ENE = wind_compass_dirs == 'ENE'
        E   = wind_compass_dirs == 'E'
        ESE = wind_compass_dirs == 'ESE'
        SE  = wind_compass_dirs == 'SE'
        SSE = wind_compass_dirs == 'SSE'
        S   = wind_compass_dirs == 'S'
        SSW = wind_compass_dirs == 'SSW'
        SW  = wind_compass_dirs == 'SW'
        WSW = wind_compass_dirs == 'WSW'
        W   = wind_compass_dirs == 'W'
        WNW = wind_compass_dirs == 'WNW'
        NW  = wind_compass_dirs == 'NW'
        NNW = wind_compass_dirs == 'NNW'

        # These are 0 m/s wind speeds. Set to 0 degrees wind direction.
        idx1 = wind_compass_dirs == '---'

        # Assign strangely labeled directions to 0 degrees
        idx2 = ((N | NNE | NE | ENE | E | ESE | SE | SSE | S | SSW | SW
                 | WSW | W | WNW | NW | NNW | idx1)==0)

        if (N | NNE | NE | ENE | E | ESE | SE | SSE | S | SSW | SW | WSW | W
            | WNW | NW | NNW | idx1 | idx2).sum() != len(wind_compass_dirs):
            ValueError('Some wind directions not accounted for!')

        wind_dir_deg[N]    = 0.0
        wind_dir_deg[NNE]  = 22.5
        wind_dir_deg[NE]   = 45.0
        wind_dir_deg[ENE]  = 67.5
        wind_dir_deg[E]    = 90.0
        wind_dir_deg[ESE]  = 112.5
        wind_dir_deg[SE]   = 135.0
        wind_dir_deg[SSE]  = 157.5
        wind_dir_deg[S]    = 180.0
        wind_dir_deg[SSW]  = 202.5
        wind_dir_deg[SW]   = 225.0
        wind_dir_deg[WSW]  = 247.5
        wind_dir_deg[W]    = 270.0
        wind_dir_deg[WNW]  = 292.5
        wind_dir_deg[NW]   = 315.0
        wind_dir_deg[NNW]  = 337.5
        wind_dir_deg[idx1] = 0.0
        wind_dir_deg[idx2] = 0.0
        return pd.Series(wind_dir_deg)

    # Define function to convert wind speed and wind direction to u&v wind components
    def windSpeedDir2uv(wind_spd, wind_dir):
        u_wind = -wind_spd * np.sin(wind_dir * np.pi /180) # Positive u-wind represents wind blowing to the east (westerly wind).
        v_wind = -wind_spd * np.cos(wind_dir * np.pi /180) # Positive v-wind represents wind blowing to the north (southerly wind).

        return u_wind, v_wind

    # Convert wind compass nominal directions to degrees
    wind_dir_deg = windCompass2Deg(wind_dir)
    u_wind, v_wind = windSpeedDir2uv(wind_speed, wind_dir_deg)

    wind_dir_deg_hi = windCompass2Deg(wind_dir_hi)
    u_wind_hi, v_wind_hi = windSpeedDir2uv(wind_speed_hi, wind_dir_deg_hi)

    # Create pandas DataFrame to hold weather data
    d = {'DATE': date_time,
      'TimeZone': time_zone,
      'temp': temp_out,
      'press': press,
      'rh': hum_out,
      'dp': dew_pt,
      'solar_irr': solar_irr,
      'wind_speed': wind_speed,
      'wind_dir': wind_dir,
      'wind_dir_deg': wind_dir_deg,
      'u_wind': u_wind,
      'v_wind': v_wind,
      'wind_speed_hi': wind_speed_hi,
      'wind_dir_hi': wind_dir_hi,
      'wind_dir_deg_hi': wind_dir_deg_hi,
      'u_wind_hi': u_wind_hi,
      'v_wind_hi': v_wind_hi,
      'air_density': air_density_in,
      'uv_index': UV_index,
      'rain': rain,
      'rain_rate': rain_rate,
      'ET': ET,
      }
    return pd.DataFrame(d)


def extract_DELTA_summary_data_single_dir(path, img_thresh=50, cn2_thresh=50):
    cn2=np.array([]); talo=cn2; img_conf=cn2; cn2_conf=cn2; IFOV=cn2;
    wind_vEst = cn2; wind_vMean = cn2;
    
    files = os.listdir(path)
    for file in files:
        if file.endswith(".h5"):
            try:
                f = h5py.File(os.path.join(path, file), 'r')
                cn2 = np.append(cn2, f['Atm']['Cn2'][:].mean(axis=1))
                talo = np.append(talo, f['G']['TALO'][:])
                img_conf = np.append(
                    img_conf, f['QCStats']['Img_Confidence'][:])
                cn2_conf = np.append(
                    cn2_conf, f['QCStats']['Cn2_Confidence'][:])
                IFOV = np.append(IFOV, f['P']['IFOV'][:])
                wind_vEst = np.append(wind_vEst, f['Wind']['vEst'][:])
                wind_vMean = np.append(wind_vMean, f['Wind']['vMean'][:])
                f.close()
            except:
                print("Failed to read data from {}.".format(file))
                pass
    
    dn = np.array([datetime.fromordinal(int(x)) + timedelta(days=x%1) \
                   - timedelta(days=366) for x in talo])
    dn = pd.Series(dn).round('1s')
    
    # cn2 = cn2 * ((2.89e-6/IFOV)**2) # scale cn2 data based on "correct" IFOV
    
    d = {
        'DATE': dn,
        'cn2': cn2,
        'log10_cn2': np.log10(cn2),
        'img_conf': img_conf,
        'cn2_conf': cn2_conf,
        'wind_vEst': wind_vEst,
        'wind_vMean': wind_vMean,
        }
    df = pd.DataFrame(d)
    df = df.loc[np.logical_and(df['img_conf']>=img_thresh, df['cn2_conf']>=cn2_thresh)].reset_index(drop=True)
    return df

def window_average(x, t, win_hms, int_hms):
    if type(x) is not pd.core.series.Series:
        raise Exception("x must be a pandas Series!")
        
    if type(t) is not pd.core.series.Series:
        raise Exception("t must be a pandas Series!")
        
    if type(t.iloc[0]) is not pd._libs.tslibs.timestamps.Timestamp:
            raise Exception("Values in t must be timestamps!")
            
    if type(win_hms) is not list:
        raise Exception("win_hms must be a 3-element list!")
        
    if type(int_hms) is not list:
        raise Exception("int_hms must be a 3-element list!")
    
    window = timedelta(
        hours=win_hms[0],
        minutes=win_hms[1],
        seconds=win_hms[2]
        )

    interval = timedelta(
        hours=int_hms[0],
        minutes=int_hms[1],
        seconds=int_hms[2]
        )
    
    x_win_avg = np.array([])
    t_win_avg = np.array([])
    
    time_left = t.iloc[0]-(window/2)
    time_right = time_left
    time_end = t.iloc[-1]
    while time_right<time_end:
        time_right = time_left + window
        idx = (t.ge(time_left) & t.lt(time_right))      # Get indices within the window
        if idx.sum()>1:
            x_avg = x.loc[idx].mean()
            t_avg = pd.Series([time_left, time_right]).mean()
            x_win_avg = np.append(x_win_avg, x_avg)
            t_win_avg = np.append(t_win_avg, t_avg)
        elif idx.sum()==1:
            x_avg = x.loc[idx]
            t_avg = t.loc[idx]
            x_win_avg = np.append(x_win_avg, x_avg) # Write x-avg to variable
            t_win_avg = np.append(t_win_avg, t_avg.iloc[0]) # Write t-avg to variable
            
        time_left = time_left+interval # Add interval to time_left
        
    t_unique, idx_unique = np.unique(t_win_avg, return_index=True) # Get unique times
    x_unique = x_win_avg[idx_unique]
    x_win_avg = pd.Series(x_unique)
    t_win_avg = pd.Series(t_unique)
    
    d = {'t_win_avg': t_win_avg, 'x_win_avg': x_win_avg}

    return pd.DataFrame(d)