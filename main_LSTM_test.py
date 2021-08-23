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
        
# %%
input_size = sequences_train_tensor.shape[-1] # 6
output_size = forecasts_train_tensor.shape[-1]

net = RNN(input_size=input_size, output_size = output_size,
          RNN_layer_size=hidden_size, num_RNN_layers=num_layers)
net.to(device, dtype)
net.train()

criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer=optimizer, step_size=STEP_SIZE, gamma=0.1)

loss_array_train = np.array([])
loss_array_test = np.array([])

for i in range(NUM_EPOCHS):
    for x, y in loader:
        # move data to proper device (and dtype if necessary)
        x = x.to(device, dtype)
        y = y.to(device, dtype)

        # zero the parameter gradients
        optimizer.zero_grad()

        out = net(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
    
    out_train = net(sequences_train_tensor.to(device, dtype))
    out_train = out_train.cpu().detach().numpy()
    out_train = out_train * (forecasts_train_max - forecasts_train_min) + forecasts_train_min
    loss_train = np.sqrt(np.mean((out_train - forecasts_train)**2))
    
    out_test = net(sequences_test_tensor.to(device, dtype))
    out_test = out_test.cpu().detach().numpy()
    out_test = out_test * (forecasts_train_max - forecasts_train_min) + forecasts_train_min
    loss_test = np.sqrt(np.mean((out_test - forecasts_test)**2))
    
    scheduler.step()

    loss_array_train = np.append(loss_array_train, loss_train)
    loss_array_test = np.append(loss_array_test, loss_test)
    print('epoch: {}; train loss: {:.4f}; test loss: {:.4f}'\
          .format(i, loss_train, loss_test))

# Plot training and validation loss curves
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(loss_array_train)+1), loss_array_train, label="train loss (last = {:.3f})".format(loss_array_train[-1]))
plt.plot(range(1, len(loss_array_test)+1), loss_array_test, label="test loss (last = {:.3f})".format(loss_array_test[-1]))
plt.ylim(0.1, 0.4)
# plt.gca().set_ylim(bottom=0)
plt.title("Model Losses")
plt.xlabel("Epoch")
plt.ylabel("$log_{10}(C_{n}^{2})$ RMSE loss")
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()

# %%
out_train = net(sequences_train_tensor.to(device, dtype))
out_train = out_train.cpu().detach().numpy()
out_train = out_train * (forecasts_train_max - forecasts_train_min) + forecasts_train_min

out_test = net(sequences_test_tensor.to(device, dtype))
out_test = out_test.cpu().detach().numpy()
out_test = out_test * (forecasts_train_max - forecasts_train_min) + forecasts_train_min

# %%
forecasts_test_df = d.get("forecasts_test_df")
forecasts_test_dts = np.array([x.index.to_pydatetime() for x in forecasts_test_df])
forecasts_test_first_dts = forecasts_test_dts[:,0]
forecasts_test_first_ds = np.array([x.date() for x in forecasts_test_first_dts])
ds_unique = np.unique(forecasts_test_first_ds)

for d0 in ds_unique:
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
    plt.xlabel('local time (EDT)')
    plt.ylabel('$C_{n}^{2} (m^{-2/3})$')
    plt.legend(loc='upper right')
    plt.xticks(rotation=30)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

#%% scatter plot analysis
forecast_idx_loss = np.sqrt(np.mean((forecasts_test - out_test)**2, axis=0))
plt.figure()
plt.plot(np.arange(1, len(forecast_idx_loss)+1)*0.5, forecast_idx_loss, '-o')
# plt.ylim(0.05, 0.2)
plt.xlabel('forecast hours')
plt.ylabel('average $log_{10}(C_{n}^{2})$ RMSE loss')
plt.grid(True)
plt.grid(True, which='minor')
plt.tight_layout()
plt.show()

# different scatter plot for each forecast index
scatter_titles = ['0.5 Hour Forecast', '1.0 Hour Forecast',
                  '1.5 Hour Forecast', '2.0 Hour Forecast',
                  '2.5 Hour Forecast', '3.0 Hour Forecast',
                  '3.5 Hour Forecast', '4.0 Hour Forecast']
for ii in range(len(forecast_idx_loss)):
    plt.figure()
    plt.plot(10**forecasts_test[:,ii], 10**out_test[:,ii],
             'g.', label='LSTM (RMSE = {:.3f})'.format(forecast_idx_loss[ii]))
    plt.plot(10**forecasts_test[:,ii], 10**forecasts_test[:,ii],
             'k.', label='truth')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(1e-17, 1e-14)
    plt.ylim(1e-17, 1e-14)
    plt.title(scatter_titles[ii])
    plt.grid(True)
    plt.grid(True, which='minor')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

# # scatter plot with all the forecast indices
# plt.figure()
# for ii in range(len(forecast_idx_loss)):
#     plt.plot(10**forecasts_test[:,ii], 10**out_test[:,ii], '.', label=f'index {ii}')
# for ii in range(len(forecast_idx_loss)):
#     plt.plot(10**forecasts_test[:,ii], 10**forecasts_test[:,ii], 'k.')
# plt.xscale('log')
# plt.yscale('log')
# plt.xlim(1e-17, 1e-14)
# plt.ylim(1e-17, 1e-14)
# # plt.title('Forecast {}'.format(ii))
# plt.grid(True)
# plt.grid(True, which='minor')
# plt.legend(loc='upper left')
# plt.tight_layout()
# plt.show()

#%% get the train datetimes
forecasts_train_df = d.get("forecasts_train_df")
forecasts_train_dts = np.array([x.index.to_pydatetime() for x in forecasts_train_df])
forecasts_valid_df = d.get("forecasts_valid_df")
forecasts_valid_dts = np.array([x.index.to_pydatetime() for x in forecasts_valid_df])
forecasts_train_dts = np.concatenate((forecasts_train_dts, forecasts_valid_dts))

#%% illustrate model performance
seq_train = sequences_16hr_train[:,:,vars_keep].copy()
seq_test_tmp = sequences_16hr_test[:,:,vars_keep].copy()
cdf_percentiles = np.array([0.1, 0.5, 0.9])

# 8/9 12:00 is the best forecast in the entire test set!
day = 3
first_hour = 16
idx_analyze = np.logical_and(
    pd.Series(forecasts_train_dts[:,0]).dt.hour==first_hour,
    pd.Series(forecasts_train_dts[:,0]).dt.minute==0)
dts_train_analyze = forecasts_train_dts[idx_analyze,:]
seq_train_analyze = seq_train[idx_analyze,:,:]
truth_train_analyze = forecasts_train[idx_analyze,:]
out_train_analyze = out_train[idx_analyze,:]

dt_test_analyze = datetime(2020, 8, day, first_hour, 0, 0)
idx_test_analyze = forecasts_test_dts[:,0]==dt_test_analyze
dts_test_analyze = forecasts_test_dts[idx_test_analyze,:].squeeze()
truth_test_analyze = forecasts_test[idx_test_analyze,:].squeeze()
out_test_analyze = out_test[idx_test_analyze,:].squeeze()

truth_cumsum = np.cumsum([1/len(truth_train_analyze)]*len(truth_train_analyze))
truth_sorted4cumsum = np.sort(truth_train_analyze, axis=0)
truth_cn2_percentiles = np.empty((3, truth_sorted4cumsum.shape[-1]))
for ii in range(truth_sorted4cumsum.shape[-1]):
    truth_cn2_percentiles[:,ii] = np.interp(
        cdf_percentiles, truth_cumsum, truth_sorted4cumsum[:,ii])

out_cumsum = np.cumsum([1/len(out_train_analyze)]*len(out_train_analyze))
out_sorted4cumsum = np.sort(out_train_analyze, axis=0)
out_cn2_percentiles = np.empty((3, out_sorted4cumsum.shape[-1]))
for ii in range(out_sorted4cumsum.shape[-1]):
    out_cn2_percentiles[:,ii] = np.interp(
        cdf_percentiles, out_cumsum, out_sorted4cumsum[:,ii])

fig, ax = plt.subplots(figsize=(8, 5))
plt.plot(dts_train_analyze[0,:], 10**truth_train_analyze[0,:],
         'k.', label='train: truth all')
for ii in range(1, len(dts_train_analyze)):
    plt.plot(dts_train_analyze[0,:], 10**truth_train_analyze[ii,:], 'k.')
plt.plot(dts_train_analyze[0,:], 10**truth_cn2_percentiles[0,:],
         '-o', label='train: truth {}%'.format(int(cdf_percentiles[0]*100)))
plt.plot(dts_train_analyze[0,:], 10**truth_cn2_percentiles[1,:],
         '-o', label='train: truth {}%'.format(int(cdf_percentiles[1]*100)))
plt.plot(dts_train_analyze[0,:], 10**truth_cn2_percentiles[2,:],
         '-o', label='train: truth {}%'.format(int(cdf_percentiles[2]*100)))
plt.plot(dts_train_analyze[0,:], 10**truth_test_analyze,
         '-o', label='test: truth')
plt.plot(dts_train_analyze[0,:], 10**out_test_analyze,
         '-o', label='test: LSTM')
plt.yscale('log')
plt.xlim(datetime(2020, 6, 1, 15, 0, 0), datetime(2020, 6, 1, 23, 59, 59))
# plt.xlim(datetime(2020, 6, 1, 11, 0, 0), datetime(2020, 6, 1, 17, 0, 0))
plt.ylim(1e-17, 1e-14)
plt.title("Test Forecast: {}".format(dt_test_analyze))
plt.xlabel('local time (EDT)')
plt.ylabel('$C_{n}^{2} (m^{-2/3})$')
ax.xaxis.set_major_formatter(myFmt)
plt.grid(True)
plt.grid(True, which='minor')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(8, 5))
plt.plot(dts_train_analyze[0,:], 10**truth_train_analyze[0,:],
         'k.', label='train: truth all')
for ii in range(1, len(dts_train_analyze)):
    plt.plot(dts_train_analyze[0,:], 10**truth_train_analyze[ii,:], 'k.')
plt.plot(dts_train_analyze[0,:], 10**truth_cn2_percentiles[0,:],
         '-o', color='tab:blue', label='train: truth {}%'.format(int(cdf_percentiles[0]*100)))
plt.plot(dts_train_analyze[0,:], 10**truth_cn2_percentiles[1,:],
         '-o', color='tab:orange', label='train: truth {}%'.format(int(cdf_percentiles[1]*100)))
plt.plot(dts_train_analyze[0,:], 10**truth_cn2_percentiles[2,:],
         '-o', color='tab:green', label='train: truth {}%'.format(int(cdf_percentiles[2]*100)))
plt.plot(dts_train_analyze[0,:], 10**out_cn2_percentiles[0,:],
         '--X', color='tab:blue', label='train: LSTM {}%'.format(int(cdf_percentiles[0]*100)))
plt.plot(dts_train_analyze[0,:], 10**out_cn2_percentiles[1,:],
         '--X', color='tab:orange', label='train: LSTM {}%'.format(int(cdf_percentiles[1]*100)))
plt.plot(dts_train_analyze[0,:], 10**out_cn2_percentiles[2,:],
         '--X', color='tab:green', label='train: LSTM {}%'.format(int(cdf_percentiles[2]*100)))
plt.plot(dts_train_analyze[0,:], 10**truth_test_analyze,
          ':P', color='blue', label='test: truth')
plt.plot(dts_train_analyze[0,:], 10**out_test_analyze,
          ':P', color='darkviolet', label='test: LSTM')
plt.yscale('log')
plt.xlim(datetime(2020, 6, 1, 15, 0, 0), datetime(2020, 6, 1, 23, 59, 59))
# plt.xlim(datetime(2020, 6, 1, 11, 0, 0), datetime(2020, 6, 1, 17, 0, 0))
plt.ylim(1e-17, 1e-14)
plt.title("Test Forecast: {}".format(dt_test_analyze))
plt.xlabel('local time (EDT)')
plt.ylabel('$C_{n}^{2} (m^{-2/3})$')
ax.xaxis.set_major_formatter(myFmt)
plt.grid(True)
plt.grid(True, which='minor')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()
