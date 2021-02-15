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

sequence_length = 8
num_layers = 1
hidden_size = 30
STEP_SIZE = 10
vars_keep = [True, False, True, False, True, True]


dtype = torch.float
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

d = pickle.load(open("dataset_formatted.pkl", "rb"))
sequences_04hr_train = d.get("sequences_04hr_train_np")
sequences_06hr_train = d.get("sequences_06hr_train_np")
sequences_08hr_train = d.get("sequences_08hr_train_np")
sequences_10hr_train = d.get("sequences_10hr_train_np")
sequences_12hr_train = d.get("sequences_12hr_train_np")
sequences_14hr_train = d.get("sequences_14hr_train_np")
sequences_16hr_train = d.get("sequences_16hr_train_np")
sequences_04hr_valid = d.get("sequences_04hr_valid_np")
sequences_06hr_valid = d.get("sequences_06hr_valid_np")
sequences_08hr_valid = d.get("sequences_08hr_valid_np")
sequences_10hr_valid = d.get("sequences_10hr_valid_np")
sequences_12hr_valid = d.get("sequences_12hr_valid_np")
sequences_14hr_valid = d.get("sequences_14hr_valid_np")
sequences_16hr_valid = d.get("sequences_16hr_valid_np")
sequences_04hr_test = d.get("sequences_04hr_test_np")
sequences_06hr_test = d.get("sequences_06hr_test_np")
sequences_08hr_test = d.get("sequences_08hr_test_np")
sequences_10hr_test = d.get("sequences_10hr_test_np")
sequences_12hr_test = d.get("sequences_12hr_test_np")
sequences_14hr_test = d.get("sequences_14hr_test_np")
sequences_16hr_test = d.get("sequences_16hr_test_np")
forecasts_train = d.get("forecasts_train_np")
forecasts_valid = d.get("forecasts_valid_np")
forecasts_test = d.get("forecasts_test_np")
sequences_train_min = d.get("sequences_train_min")
sequences_train_max = d.get("sequences_train_max")
forecasts_train_min = d.get("forecasts_train_min")
forecasts_train_max = d.get("forecasts_train_max")

# normalize
sequences_04hr_train_norm = (sequences_04hr_train - sequences_train_min) / (sequences_train_max - sequences_train_min)
sequences_06hr_train_norm = (sequences_06hr_train - sequences_train_min) / (sequences_train_max - sequences_train_min)
sequences_08hr_train_norm = (sequences_08hr_train - sequences_train_min) / (sequences_train_max - sequences_train_min)
sequences_10hr_train_norm = (sequences_10hr_train - sequences_train_min) / (sequences_train_max - sequences_train_min)
sequences_12hr_train_norm = (sequences_12hr_train - sequences_train_min) / (sequences_train_max - sequences_train_min)
sequences_14hr_train_norm = (sequences_14hr_train - sequences_train_min) / (sequences_train_max - sequences_train_min)
sequences_16hr_train_norm = (sequences_16hr_train - sequences_train_min) / (sequences_train_max - sequences_train_min)
sequences_04hr_valid_norm = (sequences_04hr_valid - sequences_train_min) / (sequences_train_max - sequences_train_min)
sequences_06hr_valid_norm = (sequences_06hr_valid - sequences_train_min) / (sequences_train_max - sequences_train_min)
sequences_08hr_valid_norm = (sequences_08hr_valid - sequences_train_min) / (sequences_train_max - sequences_train_min)
sequences_10hr_valid_norm = (sequences_10hr_valid - sequences_train_min) / (sequences_train_max - sequences_train_min)
sequences_12hr_valid_norm = (sequences_12hr_valid - sequences_train_min) / (sequences_train_max - sequences_train_min)
sequences_14hr_valid_norm = (sequences_14hr_valid - sequences_train_min) / (sequences_train_max - sequences_train_min)
sequences_16hr_valid_norm = (sequences_16hr_valid - sequences_train_min) / (sequences_train_max - sequences_train_min)
forecasts_train_norm = (forecasts_train - forecasts_train_min) / (forecasts_train_max - forecasts_train_min)

sequences_04hr_train_tensor = torch.tensor(sequences_04hr_train_norm[:,:,vars_keep], dtype=dtype)
sequences_06hr_train_tensor = torch.tensor(sequences_06hr_train_norm[:,:,vars_keep], dtype=dtype)
sequences_08hr_train_tensor = torch.tensor(sequences_08hr_train_norm[:,:,vars_keep], dtype=dtype)
sequences_10hr_train_tensor = torch.tensor(sequences_10hr_train_norm[:,:,vars_keep], dtype=dtype)
sequences_12hr_train_tensor = torch.tensor(sequences_12hr_train_norm[:,:,vars_keep], dtype=dtype)
sequences_14hr_train_tensor = torch.tensor(sequences_14hr_train_norm[:,:,vars_keep], dtype=dtype)
sequences_16hr_train_tensor = torch.tensor(sequences_16hr_train_norm[:,:,vars_keep], dtype=dtype)
sequences_04hr_valid_tensor = torch.tensor(sequences_04hr_valid_norm[:,:,vars_keep], dtype=dtype)
sequences_06hr_valid_tensor = torch.tensor(sequences_06hr_valid_norm[:,:,vars_keep], dtype=dtype)
sequences_08hr_valid_tensor = torch.tensor(sequences_08hr_valid_norm[:,:,vars_keep], dtype=dtype)
sequences_10hr_valid_tensor = torch.tensor(sequences_10hr_valid_norm[:,:,vars_keep], dtype=dtype)
sequences_12hr_valid_tensor = torch.tensor(sequences_12hr_valid_norm[:,:,vars_keep], dtype=dtype)
sequences_14hr_valid_tensor = torch.tensor(sequences_14hr_valid_norm[:,:,vars_keep], dtype=dtype)
sequences_16hr_valid_tensor = torch.tensor(sequences_16hr_valid_norm[:,:,vars_keep], dtype=dtype)
forecasts_train_tensor = torch.tensor(forecasts_train_norm, dtype=dtype)

if sequence_length==4:
    sequences_train_tensor = torch.clone(sequences_04hr_train_tensor)
    sequences_valid_tensor = torch.clone(sequences_04hr_valid_tensor)
elif sequence_length==8:
    sequences_train_tensor = torch.clone(sequences_08hr_train_tensor)
    sequences_valid_tensor = torch.clone(sequences_08hr_valid_tensor)
elif sequence_length==12:
    sequences_train_tensor = torch.clone(sequences_12hr_train_tensor)
    sequences_valid_tensor = torch.clone(sequences_12hr_valid_tensor)
elif sequence_length==16:
    sequences_train_tensor = torch.clone(sequences_16hr_train_tensor)
    sequences_valid_tensor = torch.clone(sequences_16hr_valid_tensor)

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

        self.RNN = nn.RNN(input_size=input_size, hidden_size=RNN_layer_size,
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
loss_array_valid = np.array([])

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
    
    out_valid = net(sequences_valid_tensor.to(device, dtype))
    out_valid = out_valid.cpu().detach().numpy()
    out_valid = out_valid * (forecasts_train_max - forecasts_train_min) + forecasts_train_min
    loss_valid = np.sqrt(np.mean((out_valid - forecasts_valid)**2))
    
    scheduler.step()

    loss_array_train = np.append(loss_array_train, loss_train)
    loss_array_valid = np.append(loss_array_valid, loss_valid)
    print('epoch: {}; train loss: {:.4f}; validation loss: {:.4f}'\
          .format(i, loss_train, loss_valid))

# Plot training and validation loss curves
plt.figure()
plt.plot(range(1, len(loss_array_train)+1), loss_array_train, label="train loss (last = {:.3f})".format(loss_array_train[-1]))
plt.plot(range(1, len(loss_array_valid)+1), loss_array_valid, label="validation loss (last = {:.3f})".format(loss_array_valid[-1]))
plt.ylim(0.1, 0.4)
# plt.gca().set_ylim(bottom=0)
plt.title("Model Losses")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()

# %%
out_train = net(sequences_train_tensor.to(device, dtype))
out_train = out_train.cpu().detach().numpy()
out_train = out_train * (forecasts_train_max - forecasts_train_min) + forecasts_train_min

out_valid = net(sequences_valid_tensor.to(device, dtype))
out_valid = out_valid.cpu().detach().numpy()
out_valid = out_valid * (forecasts_train_max - forecasts_train_min) + forecasts_train_min

# %%
forecasts_valid_df = d.get("forecasts_valid_df")
forecasts_valid_dts = np.array([x.index.to_pydatetime() for x in forecasts_valid_df])
forecasts_valid_first_dts = forecasts_valid_dts[:,0]
forecasts_valid_first_ds = np.array([x.date() for x in forecasts_valid_first_dts])
ds_unique = np.unique(forecasts_valid_first_ds)

for d0 in ds_unique:
    d0_idx = np.logical_and(forecasts_valid_first_ds == d0,
                            pd.Series(forecasts_valid_first_dts).dt.minute==0)
    
    dts_plot = forecasts_valid_dts[d0_idx,:]
    cn2_truth = forecasts_valid[d0_idx,:]
    cn2_forecast = out_valid[d0_idx,:]
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
