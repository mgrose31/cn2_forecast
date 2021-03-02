# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 16:35:06 2021

@author: Mitchell Grose
"""
import os
import pickle

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
myFmt = DateFormatter('%H:%M')

from datetime import date, time, datetime, timedelta

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from scipy import stats

NUM_NETS = 10

NUM_EPOCHS = 50
BATCH_SIZE = 32
lr = 0.01
wd = 1e-3

sequence_length = 16
num_layers = 2
hidden_size = 40
STEP_SIZE = 10
vars_keep = [False, True, True, False, True, True]

dtype = torch.float
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

d = pickle.load(open("dataset_formatted.pkl", "rb"))
sequences_04hr_train = d.get("sequences_04hr_train_np")
sequences_08hr_train = d.get("sequences_08hr_train_np")
sequences_12hr_train = d.get("sequences_12hr_train_np")
sequences_16hr_train = d.get("sequences_16hr_train_np")
sequences_04hr_valid = d.get("sequences_04hr_valid_np")
sequences_08hr_valid = d.get("sequences_08hr_valid_np")
sequences_12hr_valid = d.get("sequences_12hr_valid_np")
sequences_16hr_valid = d.get("sequences_16hr_valid_np")
sequences_04hr_test = d.get("sequences_04hr_test_np")
sequences_08hr_test = d.get("sequences_08hr_test_np")
sequences_12hr_test = d.get("sequences_12hr_test_np")
sequences_16hr_test = d.get("sequences_16hr_test_np")
forecasts_train = d.get("forecasts_train_np")
forecasts_valid = d.get("forecasts_valid_np")
forecasts_test = d.get("forecasts_test_np")

# combine train/validation datasets; get training min/max
sequences_04hr_train = np.concatenate((sequences_04hr_train, sequences_04hr_valid))
sequences_08hr_train = np.concatenate((sequences_08hr_train, sequences_08hr_valid))
sequences_12hr_train = np.concatenate((sequences_12hr_train, sequences_12hr_valid))
sequences_16hr_train = np.concatenate((sequences_16hr_train, sequences_16hr_valid))
forecasts_train = np.concatenate((forecasts_train, forecasts_valid))
sequences_train_min = sequences_16hr_train.min(axis=(0, 1))
sequences_train_max = sequences_16hr_train.max(axis=(0, 1))
forecasts_train_min = forecasts_train.min()
forecasts_train_max = forecasts_train.max()

# normalize
sequences_04hr_train_norm = (sequences_04hr_train - sequences_train_min) / (sequences_train_max - sequences_train_min)
sequences_08hr_train_norm = (sequences_08hr_train - sequences_train_min) / (sequences_train_max - sequences_train_min)
sequences_12hr_train_norm = (sequences_12hr_train - sequences_train_min) / (sequences_train_max - sequences_train_min)
sequences_16hr_train_norm = (sequences_16hr_train - sequences_train_min) / (sequences_train_max - sequences_train_min)
sequences_04hr_test_norm = (sequences_04hr_test - sequences_train_min) / (sequences_train_max - sequences_train_min)
sequences_08hr_test_norm = (sequences_08hr_test - sequences_train_min) / (sequences_train_max - sequences_train_min)
sequences_12hr_test_norm = (sequences_12hr_test - sequences_train_min) / (sequences_train_max - sequences_train_min)
sequences_16hr_test_norm = (sequences_16hr_test - sequences_train_min) / (sequences_train_max - sequences_train_min)
forecasts_train_norm = (forecasts_train - forecasts_train_min) / (forecasts_train_max - forecasts_train_min)

# convert to PyTorch tensors
sequences_04hr_train_tensor = torch.tensor(sequences_04hr_train_norm[:,:,vars_keep], dtype=dtype)
sequences_08hr_train_tensor = torch.tensor(sequences_08hr_train_norm[:,:,vars_keep], dtype=dtype)
sequences_12hr_train_tensor = torch.tensor(sequences_12hr_train_norm[:,:,vars_keep], dtype=dtype)
sequences_16hr_train_tensor = torch.tensor(sequences_16hr_train_norm[:,:,vars_keep], dtype=dtype)
sequences_04hr_test_tensor = torch.tensor(sequences_04hr_test_norm[:,:,vars_keep], dtype=dtype)
sequences_08hr_test_tensor = torch.tensor(sequences_08hr_test_norm[:,:,vars_keep], dtype=dtype)
sequences_12hr_test_tensor = torch.tensor(sequences_12hr_test_norm[:,:,vars_keep], dtype=dtype)
sequences_16hr_test_tensor = torch.tensor(sequences_16hr_test_norm[:,:,vars_keep], dtype=dtype)
forecasts_train_tensor = torch.tensor(forecasts_train_norm, dtype=dtype)

if sequence_length==4:
    sequences_train_tensor = torch.clone(sequences_04hr_train_tensor)
    sequences_test_tensor = torch.clone(sequences_04hr_test_tensor)
elif sequence_length==8:
    sequences_train_tensor = torch.clone(sequences_08hr_train_tensor)
    sequences_test_tensor = torch.clone(sequences_08hr_test_tensor)
elif sequence_length==12:
    sequences_train_tensor = torch.clone(sequences_12hr_train_tensor)
    sequences_test_tensor = torch.clone(sequences_12hr_test_tensor)
elif sequence_length==16:
    sequences_train_tensor = torch.clone(sequences_16hr_train_tensor)
    sequences_test_tensor = torch.clone(sequences_16hr_test_tensor)

dataset = TensorDataset(sequences_train_tensor, forecasts_train_tensor)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

forecasts_train_df = d.get("forecasts_train_df")
forecasts_train_dts = np.array([x.index.to_pydatetime() for x in forecasts_train_df])
forecasts_valid_df = d.get("forecasts_valid_df")
forecasts_valid_dts = np.array([x.index.to_pydatetime() for x in forecasts_valid_df])
forecasts_train_dts = np.concatenate((forecasts_train_dts, forecasts_valid_dts))

forecasts_test_df = d.get("forecasts_test_df")
forecasts_test_dts = np.array([x.index.to_pydatetime() for x in forecasts_test_df])
forecasts_test_first_dts = forecasts_test_dts[:,0]
forecasts_test_first_ds = np.array([x.date() for x in forecasts_test_first_dts])
ds_unique = np.unique(forecasts_test_first_ds)

# %% define RNN architecture
class RNN(nn.Module):
    def __init__(self, input_size=4, RNN_layer_size=50,
                 output_size=16, num_RNN_layers=1):
        super().__init__()
        self.input_size = input_size
        self.RNN_layer_size = RNN_layer_size
        self.output_size = output_size
        self.num_RNN_layers = num_RNN_layers

        self.RNN = nn.LSTM(input_size=input_size, hidden_size=RNN_layer_size,
                           num_layers=num_RNN_layers, batch_first=True)
        self.fc1 = nn.Linear(RNN_layer_size, output_size)

    def forward(self, x):
        x, hn = self.RNN(x)
        x = self.fc1(x)

        return x[:,-1,:] # just want last time step hidden states
        
#%%
input_size = sequences_train_tensor.shape[-1]
output_size = forecasts_train_tensor.shape[-1]

# loss arrays for training (0) and each forecast's loss (1)
loss_array0_train = np.empty((NUM_NETS, NUM_EPOCHS))
loss_array0_test = np.empty((NUM_NETS, NUM_EPOCHS))
loss_array1_train = np.empty((NUM_NETS, len(forecasts_train)))
loss_array1_test = np.empty((NUM_NETS, len(forecasts_test)))
rmse_array = np.empty((NUM_NETS, len(ds_unique)))
for hh in range(NUM_NETS):
    print("Training model {} of {}.".format(hh+1, NUM_NETS))
    net = RNN(input_size=input_size, output_size = output_size,
              RNN_layer_size=hidden_size, num_RNN_layers=num_layers)
    net.to(device, dtype)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer=optimizer, step_size=STEP_SIZE, gamma=0.1)
    
    for i in range(NUM_EPOCHS):
        net.train()
        for x, y in loader:
            x = x.to(device, dtype)
            y = y.to(device, dtype)
    
            optimizer.zero_grad()
            out = net(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
        
        net.eval()
        out_train = net(sequences_train_tensor.to(device, dtype))
        out_train = out_train.cpu().detach().numpy()
        out_train = out_train * (forecasts_train_max - forecasts_train_min) + forecasts_train_min
        loss_train = np.sqrt(np.mean((out_train - forecasts_train)**2))
        
        out_test = net(sequences_test_tensor.to(device, dtype))
        out_test = out_test.cpu().detach().numpy()
        out_test = out_test * (forecasts_train_max - forecasts_train_min) + forecasts_train_min
        loss_test = np.sqrt(np.mean((out_test - forecasts_test)**2))
        
        loss_array0_train[hh,i] = loss_train
        loss_array0_test[hh,i] = loss_test
            
        scheduler.step()

    loss_array1_train[hh,:] = np.sqrt(np.mean((out_train - forecasts_train)**2, axis=1))
    loss_array1_test[hh,:] = np.sqrt(np.mean((out_test - forecasts_test)**2, axis=1))
    
    # calculate each day's loss score
    for j, d0 in enumerate(ds_unique):
        d0_idx = forecasts_test_first_ds==d0
        cn2_truth = forecasts_test[d0_idx,:]
        cn2_forecast = out_test[d0_idx,:]
        rmse_array[hh,j] = np.sqrt(np.mean((cn2_truth - cn2_forecast)**2))
    
    d0 = ds_unique[-2]
    d0_idx = np.logical_and(forecasts_test_first_ds == d0,
                            pd.Series(forecasts_test_first_dts).dt.minute==0)
    
    dts_plot = forecasts_test_dts[d0_idx,:]
    cn2_truth = forecasts_test[d0_idx,:]
    cn2_forecast = out_test[d0_idx,:]
    rmses_plot = np.sqrt(np.mean((cn2_truth - cn2_forecast)**2, axis=1))
    
    # get unique times to plot so the truth curves do not overlap
    dts_truth_plot, i = np.unique(dts_plot, return_index=True)
    cn2_truth_plot = cn2_truth.flatten()[i]
    
    # plot
    hours_plot = [x[0].hour for x in dts_plot]
    cmap = plt.get_cmap('cool')
    norm = matplotlib.colors.Normalize(vmin=hours_plot[0], vmax=hours_plot[-1])
    clrs = cmap(norm(hours_plot))
    
    fig, ax = plt.subplots()
    for ii in range(len(dts_plot)):
        plt.plot(dts_plot[ii,:], 10**cn2_forecast[ii,:], '-o', color=clrs[ii,:])
    plt.plot(dts_truth_plot, 10**cn2_truth_plot, 'k-o', label='truth')
    ax.xaxis.set_major_formatter(myFmt)
    cbar = fig.colorbar(
        matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ticks=hours_plot)
    cbar_str = [
        r'{:02d} ({:.3f})'.format(h, z) for (h, z) in zip(hours_plot, rmses_plot)]
    cbar.ax.set_yticklabels(cbar_str)
    cbar.ax.set_title('Forecast (RMSE)')
    
    plt.yscale('log')
    plt.xlim(datetime.combine(d0, time(0, 0, 0)),
              datetime.combine(d0 + timedelta(days=1), time(0, 0, 0)))
    plt.ylim(1e-17, 1e-14)
    plt.title(d0.strftime('%m-%d-%Y'))
    plt.xlabel('local time')
    plt.ylabel('$C_{n}^{2} (m^{-2/3})$')
    plt.legend(loc='upper right')
    plt.xticks(rotation=30)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# plot all train/test loss curves and the average
loss_array0_train_mean = np.mean(loss_array0_train, axis=0)
loss_array0_test_mean = np.mean(loss_array0_test, axis=0)
plt.figure()
plt.plot(range(1, NUM_EPOCHS+1), loss_array0_train[0,:], color='tab:blue', label='train')
plt.plot(range(1, NUM_EPOCHS+1), loss_array0_test[0,:], color='tab:orange', label='test')
for jj in range(1, NUM_NETS):
    plt.plot(range(1, NUM_EPOCHS+1), loss_array0_train[jj,:], color='tab:blue')
    plt.plot(range(1, NUM_EPOCHS+1), loss_array0_test[jj,:], color='tab:orange')
plt.plot(range(1, NUM_EPOCHS+1), loss_array0_train_mean, 'k-',
         label='train mean (last = {:.4f})'.format(loss_array0_train_mean[-1]))
plt.plot(range(1, NUM_EPOCHS+1), loss_array0_test_mean, 'k--',
         label='test mean (last = {:.4f})'.format(loss_array0_test_mean[-1]))
plt.ylim(0.1, 0.4)
plt.title("LSTM Model Losses")
plt.xlabel("Epoch")
plt.ylabel("$log_{10}(C_{n}^{2})$ RMSE loss")
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()
plt.show()

# plot errors as a function of test day
rmse_array_mean = np.mean(rmse_array, axis=0)
rmse_array_std = np.std(rmse_array, axis=0)
rmse_array_sdom = rmse_array_std / np.sqrt(NUM_NETS)
rmse_array_min = np.min(rmse_array, axis=0)
rmse_array_max = np.max(rmse_array, axis=0)
plt.figure()
plt.errorbar(x=ds_unique, y=rmse_array_mean, yerr=rmse_array_sdom,
             fmt='k-o', label='mean w/ SDOM')
plt.plot(ds_unique, rmse_array_min, 'b:', label='min/max')
plt.plot(ds_unique, rmse_array_max, 'b:')
plt.ylim(0.1, 0.3)
plt.title("LSTM Daily Performance Summary")
plt.xlabel("test date")
plt.ylabel("$log_{10}(C_{n}^{2})$ RMSE loss")
plt.xticks(rotation=30)
plt.legend()
plt.grid(True)
plt.grid(True, which='minor')
plt.tight_layout()
plt.show()

# plt.figure()
# plt.plot(ds_unique, rmse_array_mean, 'k-o', label='mean')
# plt.plot(ds_unique, rmse_array_mean+rmse_array_std, 'r--', label='mean +/- std')
# plt.plot(ds_unique, rmse_array_mean-rmse_array_std, 'r--')
# plt.plot(ds_unique, rmse_array_min, 'b:', label='min/max')
# plt.plot(ds_unique, rmse_array_max, 'b:')
# plt.ylim(0.1, 0.3)
# plt.xticks(rotation=30)
# plt.legend()
# plt.grid(True)
# plt.grid(True, which='minor')
# plt.tight_layout()
# plt.show()

# calculate statistics for model performance on each test forecast
loss_array1_test_mean = loss_array1_test.mean(axis=0)
loss_array1_test_std = loss_array1_test.std(axis=0)
loss_array1_test_sdom = loss_array1_test_std / np.sqrt(NUM_NETS)
loss_array1_test_min = loss_array1_test.min(axis=0)
loss_array1_test_max = loss_array1_test.max(axis=0)

# set datetimes to same year/day to plot forecast errors by time-of-day
t_test = np.array([t.replace(year=1900, month=1, day=1) for t in forecasts_test_first_dts])

fig, ax = plt.subplots(figsize=(8, 5))
plt.plot(t_test, loss_array1_test[0,:], 'k.', label='all')
for i in range(1, len(loss_array1_test)):
    plt.plot(t_test, loss_array1_test[i,:], 'k.')
plt.plot(t_test, loss_array1_test_mean, 'r.', label='mean')
plt.ylim(0, 0.5)
ax.xaxis.set_major_formatter(myFmt)
plt.title("LSTM Performance Summary")
plt.xlabel("forecast first hour")
plt.ylabel("$log_{10}(C_{n}^{2})$ RMSE loss")
plt.legend(loc='upper left')
plt.grid(True)
plt.grid(True, which='minor')
plt.tight_layout()
plt.show()

# plt.close('all')

#%% statistics on forecast performance
num_show = 5
idx = -1

# copy the appropriate train/test sequences to local variables for analysis
seq_train = sequences_16hr_train[:,:,vars_keep].copy()
seq_test = sequences_16hr_test[:,:,vars_keep].copy()

# sort the individual-forecast rmse scores
loss_test_sort_idx = np.argsort(loss_array1_test_mean)
loss_test_mean_sorted = loss_array1_test_mean[loss_test_sort_idx]
dts_test_sorted = forecasts_test_dts[loss_test_sort_idx,:]
seq_test_sorted = seq_test[loss_test_sort_idx]
fcst_test_sorted = forecasts_test[loss_test_sort_idx,:]
out_test_sorted = out_test[loss_test_sort_idx,:]

# calculate the rmse between a test forecast and all train forecasts
seq_test0 = seq_test_sorted[idx,:,:]
forecasts_test0 = fcst_test_sorted[idx,:]
dts_test0 = dts_test_sorted[idx,:]
out_test0 = out_test_sorted[idx,:]
train_error = np.sqrt(np.mean((forecasts_test0 - forecasts_train)**2, axis=1))

# sort the rmses between the test forecast and train forecasts
train_error_sort_idx = np.argsort(train_error)

# get the train sequences/tests based on the sorting
seq_train_sorted = seq_train[train_error_sort_idx,:,:]
fcst_train_sorted = forecasts_train[train_error_sort_idx,:]

plt.figure()
plt.plot(10**fcst_train_sorted[0,:], 'k-o', label='train')
for i in range(1, num_show):
    plt.plot(10**fcst_train_sorted[i,:], 'k-o')
plt.plot(10**forecasts_test0, 'r-o', label='test')
plt.yscale('log')
plt.ylim(1e-17, 1e-14)
plt.title("Forecast")
# plt.xlabel("")
plt.ylabel("$log_{10}(C_{n}^{2})$")
plt.legend(loc="upper right")
plt.grid(True)
plt.grid(True, which="minor")
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(seq_train_sorted[0,:,0], 'k-o', label='train')
for i in range(1, num_show):
    plt.plot(seq_train_sorted[i,:,0], 'k-o')
plt.plot(seq_test0[:,0], 'r-o', label='test')
plt.title("Pressure Sequences")
plt.legend(loc="best")
plt.grid(True)
plt.grid(True, which="minor")
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(seq_train_sorted[0,:,1], 'k-o', label='train')
for i in range(1, num_show):
    plt.plot(seq_train_sorted[i,:,1], 'k-o')
plt.plot(seq_test0[:,1], 'r-o', label='test')
plt.ylim(0, 100)
plt.title("RH Sequences")
plt.legend(loc="best")
plt.grid(True)
plt.grid(True, which="minor")
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(seq_train_sorted[0,:,2], 'k-o', label='train')
for i in range(1, num_show):
    plt.plot(seq_train_sorted[i,:,2], 'k-o')
plt.plot(seq_test0[:,2], 'r-o', label='test')
plt.title("Solar Irradiance Sequences")
plt.legend(loc="best")
plt.grid(True)
plt.grid(True, which="minor")
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(seq_train_sorted[0,:,3], 'k-o', label='train')
for i in range(1, num_show):
    plt.plot(seq_train_sorted[i,:,3], 'k-o')
plt.plot(seq_test0[:,3], 'r-o', label='test')
plt.ylim(-17, -14)
plt.title("Turbulence Sequences")
plt.legend(loc="best")
plt.grid(True)
plt.grid(True, which="minor")
plt.tight_layout()
plt.show()

#%%
fcst_train_dts_sorted = forecasts_train_dts[train_error_sort_idx,:]
idx_neutral_event = np.any(fcst_train_sorted<=-16, axis=1)
out_train_sorted = out_train[train_error_sort_idx,:]
out_neutral_event = out_train_sorted[idx_neutral_event,:]
truth_neutral_event = fcst_train_sorted[idx_neutral_event,:]

# dts_neutral_event = fcst_train_dts_sorted[idx_neutral_event,:]
# plt.figure()
# for i in range(len(out_neutral_event)):
#     plt.plot(dts_neutral_event[i,:], 10**truth_neutral_event[i,:], 'k-o')
#     plt.plot(dts_neutral_event[i,:], 10**out_neutral_event[i,:], 'g-o')
# plt.yscale('log')
# # plt.xlim(datetime(1900, 1, 1, 0, 0, 1), datetime(1900, 1, 1, 23, 59, 59))
# plt.ylim(1e-17, 1e-14)
# plt.grid(True)
# plt.grid(True, which='minor')
# plt.legend(loc='best')
# plt.tight_layout()
# plt.show()




ff = fcst_train_dts_sorted.copy()
for i, v1 in enumerate(ff):
    for j, v2 in enumerate(v1):
        ff[i,j] = ff[i,j].replace(year=1900, month=1, day=1)

cc = ff[idx_neutral_event,:]

a = np.array([t.replace(year=1900, month=1, day=1) for t in dts_test0])

fig, ax = plt.subplots()
plt.plot(cc[0,:], 10**truth_neutral_event[0,:], 'k-o', label='train truth')
plt.plot(cc[0,:], 10**out_neutral_event[0,:], 'g-o', label='train forecast')
for i in range(1, len(out_neutral_event)):
    plt.plot(cc[i,:], 10**truth_neutral_event[i,:], 'k-o')
    plt.plot(cc[i,:], 10**out_neutral_event[i,:], 'g-o')
plt.plot(a, 10**forecasts_test0, 'k--o', label='test truth')
plt.plot(a, 10**out_test0, 'r-o', label='test forecast')
plt.yscale('log')
# plt.xlim(datetime(1900, 1, 1, 0, 0, 1), datetime(1900, 1, 1, 23, 59, 59))
plt.xlim(datetime(1900, 1, 1, 15, 0, 0), datetime(1900, 1, 1, 23, 59, 59))
plt.ylim(1e-17, 1e-14)
ax.xaxis.set_major_formatter(myFmt)
plt.title('Neutral Events')
plt.xlabel('local time (EST)')
plt.ylabel('$C_{n}^{2}$ ($m^{-2/3}$)')
plt.grid(True)
plt.grid(True, which='minor')
plt.legend(loc='best')
plt.tight_layout()
plt.show()
