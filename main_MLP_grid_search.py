# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 22:33:03 2021

@author: Mitchell Grose
"""
import pickle
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

start_time = datetime.now()

NUM_EPOCHS = 50
BATCH_SIZE = 32
# STEP_SIZE = 10
NUM_NETS = 25
lr = 0.01
wd = 1e-3

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
sequences_04hr_train_tensor = torch.flatten(torch.tensor(sequences_04hr_train_norm, dtype=dtype), 1, -1)
sequences_06hr_train_tensor = torch.flatten(torch.tensor(sequences_06hr_train_norm, dtype=dtype), 1, -1)
sequences_08hr_train_tensor = torch.flatten(torch.tensor(sequences_08hr_train_norm, dtype=dtype), 1, -1)
sequences_10hr_train_tensor = torch.flatten(torch.tensor(sequences_10hr_train_norm, dtype=dtype), 1, -1)
sequences_12hr_train_tensor = torch.flatten(torch.tensor(sequences_12hr_train_norm, dtype=dtype), 1, -1)
sequences_14hr_train_tensor = torch.flatten(torch.tensor(sequences_14hr_train_norm, dtype=dtype), 1, -1)
sequences_16hr_train_tensor = torch.flatten(torch.tensor(sequences_16hr_train_norm, dtype=dtype), 1, -1)
sequences_04hr_valid_tensor = torch.flatten(torch.tensor(sequences_04hr_valid_norm, dtype=dtype), 1, -1)
sequences_06hr_valid_tensor = torch.flatten(torch.tensor(sequences_06hr_valid_norm, dtype=dtype), 1, -1)
sequences_08hr_valid_tensor = torch.flatten(torch.tensor(sequences_08hr_valid_norm, dtype=dtype), 1, -1)
sequences_10hr_valid_tensor = torch.flatten(torch.tensor(sequences_10hr_valid_norm, dtype=dtype), 1, -1)
sequences_12hr_valid_tensor = torch.flatten(torch.tensor(sequences_12hr_valid_norm, dtype=dtype), 1, -1)
sequences_14hr_valid_tensor = torch.flatten(torch.tensor(sequences_14hr_valid_norm, dtype=dtype), 1, -1)
sequences_16hr_valid_tensor = torch.flatten(torch.tensor(sequences_16hr_valid_norm, dtype=dtype), 1, -1)
forecasts_train_tensor = torch.tensor(forecasts_train_norm, dtype=dtype)

# define MLP architecture
class MLP(nn.Module):
    def __init__(self, input_size, output_size, num_layers, layer_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.layer_size = layer_size
        
        if self.num_layers==0:
            self.fc1 = nn.Linear(self.input_size, self.output_size)
        elif self.num_layers==1:
            self.fc1 = nn.Linear(self.input_size, self.layer_size)
            self.fc2 = nn.Linear(self.layer_size, self.output_size)
        elif self.num_layers==2:
            self.fc1 = nn.Linear(self.input_size, self.layer_size)
            self.fc2 = nn.Linear(self.layer_size, self.layer_size)
            self.fc3 = nn.Linear(self.layer_size, self.output_size)
        elif self.num_layers==3:
            self.fc1 = nn.Linear(self.input_size, self.layer_size)
            self.fc2 = nn.Linear(self.layer_size, self.layer_size)
            self.fc3 = nn.Linear(self.layer_size, self.layer_size)
            self.fc4 = nn.Linear(self.layer_size, self.output_size)

    def forward(self, x):
        if self.num_layers==0:
            x = self.fc1(x)
        elif self.num_layers==1:
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
        elif self.num_layers==2:
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
        elif self.num_layers==3:
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = torch.relu(self.fc3(x))
            x = self.fc4(x)
        return x

# input_size = sequences_04hr_train_tensor.shape[-1]
output_size = forecasts_train_tensor.shape[-1]

seq_len_array = np.array([], dtype=int)
hidden_size_array = np.array([], dtype=int)
num_layers_array = np.array([], dtype=int)
step_size_array = np.array([], dtype=int)
# for x0 in [4, 6, 8, 10, 12, 14, 16]:
for x0 in [4, 8, 12, 16]:
    for x1 in [1, 2, 3]:
        for x2 in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
                for x3 in [10, 20]:
                    seq_len_array = np.append(seq_len_array, x0)
                    num_layers_array = np.append(num_layers_array, x1)
                    hidden_size_array = np.append(hidden_size_array, x2)
                    step_size_array = np.append(step_size_array, x3)

loss_scores_train = np.zeros((len(hidden_size_array), NUM_NETS))
loss_scores_valid = np.zeros((len(hidden_size_array), NUM_NETS))
for jj in range(len(hidden_size_array)):
    seq_len = seq_len_array[jj]
    num_layers = num_layers_array[jj]
    hidden_size = hidden_size_array[jj]
    STEP_SIZE = step_size_array[jj]
    
    if seq_len==4:
        sequences_train_tensor = torch.clone(sequences_04hr_train_tensor)
        sequences_valid_tensor = torch.clone(sequences_04hr_valid_tensor)
    elif seq_len==6:
        sequences_train_tensor = torch.clone(sequences_06hr_train_tensor)
        sequences_valid_tensor = torch.clone(sequences_06hr_valid_tensor)
    elif seq_len==8:
        sequences_train_tensor = torch.clone(sequences_08hr_train_tensor)
        sequences_valid_tensor = torch.clone(sequences_08hr_valid_tensor)
    elif seq_len==10:
        sequences_train_tensor = torch.clone(sequences_10hr_train_tensor)
        sequences_valid_tensor = torch.clone(sequences_10hr_valid_tensor)
    elif seq_len==12:
        sequences_train_tensor = torch.clone(sequences_12hr_train_tensor)
        sequences_valid_tensor = torch.clone(sequences_12hr_valid_tensor)
    elif seq_len==14:
        sequences_train_tensor = torch.clone(sequences_14hr_train_tensor)
        sequences_valid_tensor = torch.clone(sequences_14hr_valid_tensor)
    elif seq_len==16:
        sequences_train_tensor = torch.clone(sequences_16hr_train_tensor)
        sequences_valid_tensor = torch.clone(sequences_16hr_valid_tensor)
    else:
        print("Something weird has happened here!")
    
    print("Processing iteration {} of {}. Layers: {}; Hidden nodes: {}"\
          .format(jj+1, len(hidden_size_array), num_layers, hidden_size))
    
    input_size = sequences_train_tensor.shape[-1]
    for kk in range(NUM_NETS):
        dataset = TensorDataset(sequences_train_tensor, forecasts_train_tensor)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        net = MLP(input_size=input_size, output_size=output_size,
                  num_layers=num_layers, layer_size=hidden_size)
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
    "seq_len_array": seq_len_array,
    "num_layers_array": num_layers_array,
    "hidden_size_array": hidden_size_array,
    "step_size_array": step_size_array,
    "loss_scores_train": loss_scores_train,
    "loss_scores_valid": loss_scores_valid,
    }
pickle.dump(d_save, open("results_grid_search_MLP.pkl", "wb"))

rmse_valid_mean = loss_scores_valid.mean(axis=1)
rmse_valid_mean_min = rmse_valid_mean.min()
rmse_valid_mean_min_idx = rmse_valid_mean.argmin()

print("Best validation dataset loss score is {:.5f}".format(rmse_valid_mean_min))
print("Best sequence length: {}".format(seq_len_array[rmse_valid_mean_min_idx]))
print("Best number of layers: {}".format(num_layers_array[rmse_valid_mean_min_idx]))
print("Best number of nodes: {}".format(hidden_size_array[rmse_valid_mean_min_idx]))
print("Best step size: {}".format(step_size_array[rmse_valid_mean_min_idx]))

print("Total elapsed time is {}".format(datetime.now() - start_time))