# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 19:34:22 2021

@author: Mitchell Grose
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
# plt.style.use('default')

from scipy import stats

d_MLP = pickle.load(open(r"results_grid_search_MLP.pkl", "rb"))
d_RNN = pickle.load(open(r"results_grid_search_RNN.pkl", "rb"))
d_GRU = pickle.load(open(r"results_grid_search_GRU.pkl", "rb"))
d_LSTM = pickle.load(open(r"results_grid_search_LSTM.pkl", "rb"))

num_nets = d_GRU.get('num_nets')
binary_array = d_GRU.get('binary_array')
seq_len_array = d_GRU.get("seq_len_array")
hidden_size_array = d_GRU.get('hidden_size_array')
num_layers_array = d_GRU.get('num_layers_array')
step_size_array = d_GRU.get('step_size_array')

rmse_train_MLP = d_MLP.get('loss_scores_train')
rmse_train_RNN = d_RNN.get('loss_scores_train')
rmse_train_GRU = d_GRU.get('loss_scores_train')
rmse_train_LSTM = d_LSTM.get('loss_scores_train')

rmse_valid_MLP = d_MLP.get('loss_scores_valid')
rmse_valid_RNN = d_RNN.get('loss_scores_valid')
rmse_valid_GRU = d_GRU.get('loss_scores_valid')
rmse_valid_LSTM = d_LSTM.get('loss_scores_valid')

rmse_train_MLP_mean = np.mean(rmse_train_MLP, axis=1)
rmse_train_RNN_mean = np.mean(rmse_train_RNN, axis=1)
rmse_train_GRU_mean = np.mean(rmse_train_GRU, axis=1)
rmse_train_LSTM_mean = np.mean(rmse_train_LSTM, axis=1)

rmse_valid_MLP_mean = np.mean(rmse_valid_MLP, axis=1)
rmse_valid_RNN_mean = np.mean(rmse_valid_RNN, axis=1)
rmse_valid_GRU_mean = np.mean(rmse_valid_GRU, axis=1)
rmse_valid_LSTM_mean = np.mean(rmse_valid_LSTM, axis=1)

rmse_train_MLP_std = np.std(rmse_train_MLP, axis=1)
rmse_train_RNN_std = np.std(rmse_train_RNN, axis=1)
rmse_train_GRU_std = np.std(rmse_train_GRU, axis=1)
rmse_train_LSTM_std = np.std(rmse_train_LSTM, axis=1)

rmse_valid_MLP_std = np.std(rmse_valid_MLP, axis=1)
rmse_valid_RNN_std = np.std(rmse_valid_RNN, axis=1)
rmse_valid_GRU_std = np.std(rmse_valid_GRU, axis=1)
rmse_valid_LSTM_std = np.std(rmse_valid_LSTM, axis=1)

rmse_train_MLP_sdom = rmse_train_MLP_std / np.sqrt(num_nets)
rmse_train_RNN_sdom = rmse_train_RNN_std / np.sqrt(num_nets)
rmse_train_GRU_sdom = rmse_train_GRU_std / np.sqrt(num_nets)
rmse_train_LSTM_sdom = rmse_train_LSTM_std / np.sqrt(num_nets)

rmse_valid_MLP_sdom = rmse_valid_MLP_std / np.sqrt(num_nets)
rmse_valid_RNN_sdom = rmse_valid_RNN_std / np.sqrt(num_nets)
rmse_valid_GRU_sdom = rmse_valid_GRU_std / np.sqrt(num_nets)
rmse_valid_LSTM_sdom = rmse_valid_LSTM_std / np.sqrt(num_nets)

rmse_train_MLP_min = np.min(rmse_train_MLP, axis=1)
rmse_train_RNN_min = np.min(rmse_train_RNN, axis=1)
rmse_train_GRU_min = np.min(rmse_train_GRU, axis=1)
rmse_train_LSTM_min = np.min(rmse_train_LSTM, axis=1)

rmse_valid_MLP_min = np.min(rmse_valid_MLP, axis=1)
rmse_valid_RNN_min = np.min(rmse_valid_RNN, axis=1)
rmse_valid_GRU_min = np.min(rmse_valid_GRU, axis=1)
rmse_valid_LSTM_min = np.min(rmse_valid_LSTM, axis=1)

rmse_train_MLP_max = np.max(rmse_train_MLP, axis=1)
rmse_train_RNN_max = np.max(rmse_train_RNN, axis=1)
rmse_train_GRU_max = np.max(rmse_train_GRU, axis=1)
rmse_train_LSTM_max = np.max(rmse_train_LSTM, axis=1)

rmse_valid_MLP_max = np.max(rmse_valid_MLP, axis=1)
rmse_valid_RNN_max = np.max(rmse_valid_RNN, axis=1)
rmse_valid_GRU_max = np.max(rmse_valid_GRU, axis=1)
rmse_valid_LSTM_max = np.max(rmse_valid_LSTM, axis=1)

# %% sort results based on average validation loss score
idx_sort_MLP = np.argsort(rmse_valid_MLP_mean)
rmse_valid_MLP_sorted = rmse_valid_MLP[idx_sort_MLP,:]
rmse_valid_mean_MLP_sorted = rmse_valid_MLP_mean[idx_sort_MLP]
rmse_train_mean_MLP_sorted = rmse_train_MLP_mean[idx_sort_MLP]
rmse_valid_sdom_MLP_sorted = rmse_valid_MLP_sdom[idx_sort_MLP]
binary_array_MLP_sorted = binary_array[idx_sort_MLP,:]
seq_len_array_MLP_sorted = seq_len_array[idx_sort_MLP]
num_layers_array_MLP_sorted = num_layers_array[idx_sort_MLP]
hidden_size_array_MLP_sorted = hidden_size_array[idx_sort_MLP]
step_size_array_MLP_sorted = step_size_array[idx_sort_MLP]

# print("Results for MLP:")
# print("Average best validation dataset loss score is {:.5f}".format(rmse_valid_mean_MLP_sorted[0]))
# print("Average best validation dataset binary array is {}".format(binary_array_MLP_sorted[0,:]))
# print("Average best sequence length: {}".format(seq_len_array_MLP_sorted[0]))
# print("Average best number of layers: {}".format(num_layers_array_MLP_sorted[0]))
# print("Average best number of nodes: {}".format(hidden_size_array_MLP_sorted[0]))
# print("Average best step size: {}".format(step_size_array_MLP_sorted[0]))


idx_sort_GRU = np.argsort(rmse_valid_GRU_mean)
rmse_valid_GRU_sorted = rmse_valid_GRU[idx_sort_GRU,:]
rmse_valid_mean_GRU_sorted = rmse_valid_GRU_mean[idx_sort_GRU]
rmse_train_mean_GRU_sorted = rmse_train_GRU_mean[idx_sort_GRU]
rmse_valid_sdom_GRU_sorted = rmse_valid_GRU_sdom[idx_sort_GRU]
binary_array_GRU_sorted = binary_array[idx_sort_GRU,:]
seq_len_array_GRU_sorted = seq_len_array[idx_sort_GRU]
num_layers_array_GRU_sorted = num_layers_array[idx_sort_GRU]
hidden_size_array_GRU_sorted = hidden_size_array[idx_sort_GRU]
step_size_array_GRU_sorted = step_size_array[idx_sort_GRU]

# print("Results for GRU:")
# print("Average best validation dataset loss score is {:.5f}".format(rmse_valid_mean_GRU_sorted[0]))
# print("Average best validation dataset binary array is {}".format(binary_array_GRU_sorted[0,:]))
# print("Average best sequence length: {}".format(seq_len_array_GRU_sorted[0]))
# print("Average best number of layers: {}".format(num_layers_array_GRU_sorted[0]))
# print("Average best number of nodes: {}".format(hidden_size_array_GRU_sorted[0]))
# print("Average best step size: {}".format(step_size_array_GRU_sorted[0]))


idx_sort_RNN = np.argsort(rmse_valid_RNN_mean)
rmse_valid_RNN_sorted = rmse_valid_RNN[idx_sort_RNN,:]
rmse_valid_mean_RNN_sorted = rmse_valid_RNN_mean[idx_sort_RNN]
rmse_train_mean_RNN_sorted = rmse_train_RNN_mean[idx_sort_RNN]
rmse_valid_sdom_RNN_sorted = rmse_valid_RNN_sdom[idx_sort_RNN]
binary_array_RNN_sorted = binary_array[idx_sort_RNN,:]
seq_len_array_RNN_sorted = seq_len_array[idx_sort_RNN]
num_layers_array_RNN_sorted = num_layers_array[idx_sort_RNN]
hidden_size_array_RNN_sorted = hidden_size_array[idx_sort_RNN]
step_size_array_RNN_sorted = step_size_array[idx_sort_RNN]

# print("Results for RNN:")
# print("Average best validation dataset loss score is {:.5f}".format(rmse_valid_mean_RNN_sorted[0]))
# print("Average best validation dataset binary array is {}".format(binary_array_RNN_sorted[0,:]))
# print("Average best sequence length: {}".format(seq_len_array_RNN_sorted[0]))
# print("Average best number of layers: {}".format(num_layers_array_RNN_sorted[0]))
# print("Average best number of nodes: {}".format(hidden_size_array_RNN_sorted[0]))
# print("Average best step size: {}".format(step_size_array_RNN_sorted[0]))


idx_sort_LSTM = np.argsort(rmse_valid_LSTM_mean)
rmse_valid_LSTM_sorted = rmse_valid_LSTM[idx_sort_LSTM,:]
rmse_valid_mean_LSTM_sorted = rmse_valid_LSTM_mean[idx_sort_LSTM]
rmse_train_mean_LSTM_sorted = rmse_train_LSTM_mean[idx_sort_LSTM]
rmse_valid_sdom_LSTM_sorted = rmse_valid_LSTM_sdom[idx_sort_LSTM]
binary_array_LSTM_sorted = binary_array[idx_sort_LSTM,:]
seq_len_array_LSTM_sorted = seq_len_array[idx_sort_LSTM]
num_layers_array_LSTM_sorted = num_layers_array[idx_sort_LSTM]
hidden_size_array_LSTM_sorted = hidden_size_array[idx_sort_LSTM]
step_size_array_LSTM_sorted = step_size_array[idx_sort_LSTM]

# print("Results for LSTM:")
# print("Average best validation dataset loss score is {:.5f}".format(rmse_valid_mean_LSTM_sorted[0]))
# print("Average best validation dataset binary array is {}".format(binary_array_LSTM_sorted[0,:]))
# print("Average best sequence length: {}".format(seq_len_array_LSTM_sorted[0]))
# print("Average best number of layers: {}".format(num_layers_array_LSTM_sorted[0]))
# print("Average best number of nodes: {}".format(hidden_size_array_LSTM_sorted[0]))
# print("Average best step size: {}".format(step_size_array_LSTM_sorted[0]))

#%% look at individual best performing models
array_min_MLP = rmse_valid_MLP.min(axis=1)
argmin_MLP = array_min_MLP.argmin()
min_MLP = min(array_min_MLP)
array_min_GRU = rmse_valid_GRU.min(axis=1)
argmin_GRU = array_min_GRU.argmin()
array_min_RNN = rmse_valid_RNN.min(axis=1)
argmin_RNN = array_min_RNN.argmin()
array_min_LSTM = rmse_valid_LSTM.min(axis=1)
argmin_LSTM = array_min_LSTM.argmin()

#%% plot variable counts in best 10% and worst 10% performing variable sets
num_y = 88
x = ['Temp', 'Press', 'RH', 'Wind Spd',
     'Solar Irr', '$C_{n}^{2}$']
x_pos = np.array([i for i, _ in enumerate(x)])
y1_MLP = binary_array_MLP_sorted[:num_y, :].sum(axis=0)
y1_GRU = binary_array_GRU_sorted[:num_y, :].sum(axis=0)
y1_RNN = binary_array_RNN_sorted[:num_y, :].sum(axis=0)
y1_LSTM = binary_array_LSTM_sorted[:num_y, :].sum(axis=0)

y2_MLP = binary_array_MLP_sorted[-num_y:, :].sum(axis=0)
y2_GRU = binary_array_GRU_sorted[-num_y:, :].sum(axis=0)
y2_RNN = binary_array_RNN_sorted[-num_y:, :].sum(axis=0)
y2_LSTM = binary_array_LSTM_sorted[-num_y:, :].sum(axis=0)

width = 0.1
plt.figure()
plt.bar(x_pos-1.5*width, y1_MLP, width, label='MLP')
plt.bar(x_pos-0.5*width, y1_GRU, width, label='GRU')
plt.bar(x_pos+0.5*width, y1_RNN, width, label='RNN')
plt.bar(x_pos+1.5*width, y1_LSTM, width, label='LSTM')
plt.ylim(0, 90)
plt.title('Best {} (10%) Variable Sets'.format(int(num_y)))
plt.xlabel('Input Variable')
plt.ylabel('Counts')
plt.xticks(ticks=x_pos, labels=x)
plt.legend(loc='best')
plt.grid(True)
plt.grid(True, which='minor')
plt.tight_layout()
plt.show()

plt.figure()
plt.bar(x_pos-1.5*width, y2_MLP, width, label='MLP')
plt.bar(x_pos-0.5*width, y2_GRU, width, label='GRU')
plt.bar(x_pos+0.5*width, y2_RNN, width, label='RNN')
plt.bar(x_pos+1.5*width, y2_LSTM, width, label='LSTM')
plt.ylim(0, 90)
plt.title('Worst {} (10%) Variable Sets'.format(int(num_y)))
plt.xlabel('Input Variable')
plt.ylabel('Counts')
plt.xticks(ticks=x_pos, labels=x)
plt.legend(loc='best')
plt.grid(True)
plt.grid(True, which='minor')
plt.tight_layout()
plt.show()

#%% significance tests (student's t-test)
alpha = 0.05

# data_A = rmse_valid_MLP_sorted[0,:]
# data_B = np.transpose(rmse_valid_MLP_sorted[1:,:])
# data_B = np.transpose(rmse_valid_GRU_sorted[1:,:])
# data_B = np.transpose(rmse_valid_RNN_sorted[1:,:])
# data_B = np.transpose(rmse_valid_LSTM_sorted[1:,:])

data_A = rmse_valid_GRU_sorted[0,:]
# data_B = np.transpose(rmse_valid_MLP_sorted[1:,:])
data_B = np.transpose(rmse_valid_GRU_sorted[1:,:])
# data_B = np.transpose(rmse_valid_RNN_sorted[1:,:])
# data_B = np.transpose(rmse_valid_LSTM_sorted[1:,:])
t, p = stats.ttest_ind(data_A, data_B, equal_var=False)

# data_A = rmse_valid_RNN_sorted[0,:]
# data_B = np.transpose(rmse_valid_MLP_sorted[1:,:])
# data_B = np.transpose(rmse_valid_GRU_sorted[1:,:])
# data_B = np.transpose(rmse_valid_RNN_sorted[1:,:])
# data_B = np.transpose(rmse_valid_LSTM_sorted[1:,:])
# t, p = stats.ttest_ind(data_A, data_B, equal_var=False)

# data_A = rmse_valid_LSTM_sorted[0,:]
# data_B = np.transpose(rmse_valid_MLP_sorted[1:,:])
# data_B = np.transpose(rmse_valid_GRU_sorted[1:,:])
# data_B = np.transpose(rmse_valid_RNN_sorted[1:,:])
# data_B = np.transpose(rmse_valid_LSTM_sorted[1:,:])
# t, p = stats.ttest_ind(data_A, data_B, equal_var=False)

# if p > alpha:
#     print("At the 5% level, fail to reject (accept) the null hypothesis that the means are equal. p={:.5f}".format(p))
# else:
#     print("At the 5% level, reject the null hypothesis that the means are equal. p={:.5f}".format(p))
