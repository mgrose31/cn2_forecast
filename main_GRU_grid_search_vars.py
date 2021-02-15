# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 22:33:03 2021

@author: Mitchell Grose
"""
import os
import pickle
import math

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
myFmt = DateFormatter('%H:%M')

from datetime import date, time, datetime, timedelta

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

start_time = datetime.now()

NUM_EPOCHS = 50
BATCH_SIZE = 32
NUM_NETS = 10
lr = 0.01
wd = 1e-3

# STEP_SIZE = 10

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

# put data into PyTorch tensors
sequences_04hr_train_tensor = torch.tensor(sequences_04hr_train_norm, dtype=dtype)
sequences_06hr_train_tensor = torch.tensor(sequences_06hr_train_norm, dtype=dtype)
sequences_08hr_train_tensor = torch.tensor(sequences_08hr_train_norm, dtype=dtype)
sequences_10hr_train_tensor = torch.tensor(sequences_10hr_train_norm, dtype=dtype)
sequences_12hr_train_tensor = torch.tensor(sequences_12hr_train_norm, dtype=dtype)
sequences_14hr_train_tensor = torch.tensor(sequences_14hr_train_norm, dtype=dtype)
sequences_16hr_train_tensor = torch.tensor(sequences_16hr_train_norm, dtype=dtype)
sequences_04hr_valid_tensor = torch.tensor(sequences_04hr_valid_norm, dtype=dtype)
sequences_06hr_valid_tensor = torch.tensor(sequences_06hr_valid_norm, dtype=dtype)
sequences_08hr_valid_tensor = torch.tensor(sequences_08hr_valid_norm, dtype=dtype)
sequences_10hr_valid_tensor = torch.tensor(sequences_10hr_valid_norm, dtype=dtype)
sequences_12hr_valid_tensor = torch.tensor(sequences_12hr_valid_norm, dtype=dtype)
sequences_14hr_valid_tensor = torch.tensor(sequences_14hr_valid_norm, dtype=dtype)
sequences_16hr_valid_tensor = torch.tensor(sequences_16hr_valid_norm, dtype=dtype)
forecasts_train_tensor = torch.tensor(forecasts_train_norm, dtype=dtype)

# define RNN architecture
class RNN(nn.Module):
    def __init__(self, input_size=4, RNN_layer_size=50,
                  output_size=16, num_RNN_layers=1):
        super().__init__()
        self.input_size = input_size
        self.RNN_layer_size = RNN_layer_size
        self.output_size = output_size
        self.num_RNN_layers = num_RNN_layers

        self.RNN = nn.GRU(input_size=input_size, hidden_size=RNN_layer_size,
                          num_layers=num_RNN_layers, batch_first=True)
        self.fc1 = nn.Linear(RNN_layer_size, output_size)

    def forward(self, x):
        x, hn = self.RNN(x)
        x = self.fc1(x)

        return x[:,-1,:] # just want last time step hidden states

# input_size = sequences_04hr_train_tensor.shape[-1]
output_size = forecasts_train_tensor.shape[-1]

# %%
def generate_binary(n):

  bin_arr = []
  bin_str = [0] * n

  for i in range(0, int(math.pow(2,n))):

    bin_arr.append("".join(map(str,bin_str))[::-1])
    bin_str[0] += 1

    # Iterate through entire array if there carrying
    for j in range(0, len(bin_str) - 1):

      if bin_str[j] == 2:
        bin_str[j] = 0
        bin_str[j+1] += 1
        continue

      else:
        break

  return bin_arr

binary_list_tmp = generate_binary(6)

binary_list = []
for this_str in binary_list_tmp:
    b = np.array([], dtype=int)
    for this_char in this_str:
        b = np.append(b, int(this_char))
    if b.sum()>=4:
        binary_list.append(b)

binary_list = np.array(binary_list)

# %% set up the array of architecture/training combinations
# d_grid_search = pickle.load(open(r"results_grid_search_GRU.pkl", "rb"))
# seq_len_array = d_grid_search.get("seq_len_array")
# hidden_size_array = d_grid_search.get('hidden_size_array')
# num_layers_array = d_grid_search.get('num_layers_array')
# step_size_array = d_grid_search.get('step_size_array')
# rmse_train_grid_search = d_grid_search.get('loss_scores_train')
# rmse_valid_grid_search = d_grid_search.get('loss_scores_valid')
# rmse_valid_grid_search_mean = np.mean(rmse_valid_grid_search, axis=1)

# idx_sort_grid_search = np.argsort(rmse_valid_grid_search_mean)
# rmse_valid_mean_grid_search_sorted = rmse_valid_grid_search_mean[idx_sort_grid_search[:10]]
# seq_len_array_grid_search_sorted = seq_len_array[idx_sort_grid_search[:10]]
# num_layers_array_grid_search_sorted = num_layers_array[idx_sort_grid_search[:10]]
# hidden_size_array_grid_search_sorted = hidden_size_array[idx_sort_grid_search[:10]]
# step_size_array_grid_search_sorted = step_size_array[idx_sort_grid_search[:10]]

# seq_len_array = np.array([], dtype=int)
# hidden_size_array = np.array([], dtype=int)
# num_layers_array = np.array([], dtype=int)
# step_size_array = np.array([], dtype=int)
# binary_array = np.empty((0, 6))
# for x0 in binary_list:
#     for x1 in range(0, len(rmse_valid_mean_grid_search_sorted)):
#         binary_array = np.concatenate((binary_array, np.expand_dims(x0, axis=0)), axis=0)
#         seq_len_array = np.append(seq_len_array, seq_len_array_grid_search_sorted[x1])
#         num_layers_array = np.append(num_layers_array, num_layers_array_grid_search_sorted[x1])
#         hidden_size_array = np.append(hidden_size_array, hidden_size_array_grid_search_sorted[x1])
#         step_size_array = np.append(step_size_array, step_size_array_grid_search_sorted[x1])

binary_array = np.empty((0, 6))
seq_len_array = np.array([], dtype=int)
hidden_size_array = np.array([], dtype=int)
num_layers_array = np.array([], dtype=int)
step_size_array = np.array([], dtype=int)
for x0 in binary_list:
    for x1 in [4, 8, 12, 16]:
        for x2 in [1, 2]:
            for x3 in [10, 20, 30, 40, 50]:
                    for x4 in [10]:
                        binary_array = np.concatenate((binary_array, np.expand_dims(x0, axis=0)), axis=0)
                        seq_len_array = np.append(seq_len_array, x1)
                        num_layers_array = np.append(num_layers_array, x2)
                        hidden_size_array = np.append(hidden_size_array, x3)
                        step_size_array = np.append(step_size_array, x4)

# %% train
loss_scores_train = np.zeros((len(hidden_size_array), NUM_NETS))
loss_scores_valid = np.zeros((len(hidden_size_array), NUM_NETS))
for jj in range(len(hidden_size_array)):
    seq_len = seq_len_array[jj]
    num_layers = num_layers_array[jj]
    hidden_size = hidden_size_array[jj]
    STEP_SIZE = step_size_array[jj]
    keep_vars = binary_array[jj,:]==1
    
    if seq_len==4:
        sequences_train_tensor = torch.clone(sequences_04hr_train_tensor[:,:,keep_vars])
        sequences_valid_tensor = torch.clone(sequences_04hr_valid_tensor[:,:,keep_vars])
    elif seq_len==8:
        sequences_train_tensor = torch.clone(sequences_08hr_train_tensor[:,:,keep_vars])
        sequences_valid_tensor = torch.clone(sequences_08hr_valid_tensor[:,:,keep_vars])
    elif seq_len==12:
        sequences_train_tensor = torch.clone(sequences_12hr_train_tensor[:,:,keep_vars])
        sequences_valid_tensor = torch.clone(sequences_12hr_valid_tensor[:,:,keep_vars])
    elif seq_len==16:
        sequences_train_tensor = torch.clone(sequences_16hr_train_tensor[:,:,keep_vars])
        sequences_valid_tensor = torch.clone(sequences_16hr_valid_tensor[:,:,keep_vars])
    else:
        print("Something weird has happened!")
    
    input_size = sequences_train_tensor.shape[-1]
    print("Processing iteration {} of {}. Sequence Length: {}; Variables: {}; Layers: {}; Hidden nodes: {}"\
          .format(jj+1, len(hidden_size_array), seq_len, keep_vars, num_layers, hidden_size))
    for kk in range(NUM_NETS):
        dataset = TensorDataset(sequences_train_tensor, forecasts_train_tensor)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        net = RNN(input_size=input_size, output_size=output_size,
                  RNN_layer_size=hidden_size, num_RNN_layers=num_layers)
        net.to(device, dtype)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer, step_size=STEP_SIZE, gamma=0.1)
        
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
            
            scheduler.step()
        
        out_train = net(sequences_train_tensor.to(device, dtype))
        out_train = out_train.cpu().detach().numpy()
        out_train = out_train * (forecasts_train_max - forecasts_train_min) + forecasts_train_min
        loss_train_final = np.sqrt(np.mean((out_train - forecasts_train)**2))
        
        out_valid = net(sequences_valid_tensor.to(device, dtype))
        out_valid = out_valid.cpu().detach().numpy()
        out_valid = out_valid * (forecasts_train_max - forecasts_train_min) + forecasts_train_min
        loss_valid_final = np.sqrt(np.mean((out_valid - forecasts_valid)**2))
        
        loss_scores_train[jj,kk] = loss_train_final
        loss_scores_valid[jj,kk] = loss_valid_final

d_save = {
    "num_nets": NUM_NETS,
    "num_epochs": NUM_EPOCHS,
    "batch_size": BATCH_SIZE,
    "weight_decay": wd,
    "binary_array": binary_array,
    "seq_len_array": seq_len_array,
    "num_layers_array": num_layers_array,
    "hidden_size_array": hidden_size_array,
    "step_size_array": step_size_array,
    "loss_scores_train": loss_scores_train,
    "loss_scores_valid": loss_scores_valid,
    }
pickle.dump(d_save, open("results_grid_search_vars_GRU.pkl", "wb"))

rmse_valid_mean = loss_scores_valid.mean(axis=1)
rmse_valid_mean_min = rmse_valid_mean.min()
rmse_valid_mean_min_idx = rmse_valid_mean.argmin()

print("Best validation dataset loss score is {:.5f}".format(rmse_valid_mean_min))
print("Best sequence length: {}".format(seq_len_array[rmse_valid_mean_min_idx]))
print("Best number of layers: {}".format(num_layers_array[rmse_valid_mean_min_idx]))
print("Best number of nodes: {}".format(hidden_size_array[rmse_valid_mean_min_idx]))
print("Best step size: {}".format(step_size_array[rmse_valid_mean_min_idx]))
print("Best variables are: {}".format(binary_array[rmse_valid_mean_min_idx,:]))