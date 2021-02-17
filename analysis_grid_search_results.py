# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 19:34:22 2021

@author: Mitchell Grose
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
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
rmse_valid_mean_MLP_sorted = rmse_valid_MLP_mean[idx_sort_MLP]
rmse_train_mean_MLP_sorted = rmse_train_MLP_mean[idx_sort_MLP]
rmse_valid_sdom_MLP_sorted = rmse_valid_MLP_sdom[idx_sort_MLP]
binary_array_MLP_sorted = binary_array[idx_sort_MLP,:]
seq_len_array_MLP_sorted = seq_len_array[idx_sort_MLP]
num_layers_array_MLP_sorted = num_layers_array[idx_sort_MLP]
hidden_size_array_MLP_sorted = hidden_size_array[idx_sort_MLP]
step_size_array_MLP_sorted = step_size_array[idx_sort_MLP]

print("Results for MLP:")
print("Average best validation dataset loss score is {:.5f}".format(rmse_valid_mean_MLP_sorted[0]))
print("Average best validation dataset binary array is {}".format(binary_array_MLP_sorted[0,:]))
print("Average best sequence length: {}".format(seq_len_array_MLP_sorted[0]))
print("Average best number of layers: {}".format(num_layers_array_MLP_sorted[0]))
print("Average best number of nodes: {}".format(hidden_size_array_MLP_sorted[0]))
print("Average best step size: {}".format(step_size_array_MLP_sorted[0]))


idx_sort_GRU = np.argsort(rmse_valid_GRU_mean)
rmse_valid_mean_GRU_sorted = rmse_valid_GRU_mean[idx_sort_GRU]
rmse_train_mean_GRU_sorted = rmse_train_GRU_mean[idx_sort_GRU]
rmse_valid_sdom_GRU_sorted = rmse_valid_GRU_sdom[idx_sort_GRU]
binary_array_GRU_sorted = binary_array[idx_sort_GRU,:]
seq_len_array_GRU_sorted = seq_len_array[idx_sort_GRU]
num_layers_array_GRU_sorted = num_layers_array[idx_sort_GRU]
hidden_size_array_GRU_sorted = hidden_size_array[idx_sort_GRU]
step_size_array_GRU_sorted = step_size_array[idx_sort_GRU]

print("Results for GRU:")
print("Average best validation dataset loss score is {:.5f}".format(rmse_valid_mean_GRU_sorted[0]))
print("Average best validation dataset binary array is {}".format(binary_array_GRU_sorted[0,:]))
print("Average best sequence length: {}".format(seq_len_array_GRU_sorted[0]))
print("Average best number of layers: {}".format(num_layers_array_GRU_sorted[0]))
print("Average best number of nodes: {}".format(hidden_size_array_GRU_sorted[0]))
print("Average best step size: {}".format(step_size_array_GRU_sorted[0]))


idx_sort_RNN = np.argsort(rmse_valid_RNN_mean)
rmse_valid_mean_RNN_sorted = rmse_valid_RNN_mean[idx_sort_RNN]
rmse_train_mean_RNN_sorted = rmse_train_RNN_mean[idx_sort_RNN]
rmse_valid_sdom_RNN_sorted = rmse_valid_RNN_sdom[idx_sort_RNN]
binary_array_RNN_sorted = binary_array[idx_sort_RNN,:]
seq_len_array_RNN_sorted = seq_len_array[idx_sort_RNN]
num_layers_array_RNN_sorted = num_layers_array[idx_sort_RNN]
hidden_size_array_RNN_sorted = hidden_size_array[idx_sort_RNN]
step_size_array_RNN_sorted = step_size_array[idx_sort_RNN]

print("Results for RNN:")
print("Average best validation dataset loss score is {:.5f}".format(rmse_valid_mean_RNN_sorted[0]))
print("Average best validation dataset binary array is {}".format(binary_array_RNN_sorted[0,:]))
print("Average best sequence length: {}".format(seq_len_array_RNN_sorted[0]))
print("Average best number of layers: {}".format(num_layers_array_RNN_sorted[0]))
print("Average best number of nodes: {}".format(hidden_size_array_RNN_sorted[0]))
print("Average best step size: {}".format(step_size_array_RNN_sorted[0]))


idx_sort_LSTM = np.argsort(rmse_valid_LSTM_mean)
rmse_valid_mean_LSTM_sorted = rmse_valid_LSTM_mean[idx_sort_LSTM]
rmse_train_mean_LSTM_sorted = rmse_train_LSTM_mean[idx_sort_LSTM]
rmse_valid_sdom_LSTM_sorted = rmse_valid_LSTM_sdom[idx_sort_LSTM]
binary_array_LSTM_sorted = binary_array[idx_sort_LSTM,:]
seq_len_array_LSTM_sorted = seq_len_array[idx_sort_LSTM]
num_layers_array_LSTM_sorted = num_layers_array[idx_sort_LSTM]
hidden_size_array_LSTM_sorted = hidden_size_array[idx_sort_LSTM]
step_size_array_LSTM_sorted = step_size_array[idx_sort_LSTM]

print("Results for LSTM:")
print("Average best validation dataset loss score is {:.5f}".format(rmse_valid_mean_LSTM_sorted[0]))
print("Average best validation dataset binary array is {}".format(binary_array_LSTM_sorted[0,:]))
print("Average best sequence length: {}".format(seq_len_array_LSTM_sorted[0]))
print("Average best number of layers: {}".format(num_layers_array_LSTM_sorted[0]))
print("Average best number of nodes: {}".format(hidden_size_array_LSTM_sorted[0]))
print("Average best step size: {}".format(step_size_array_LSTM_sorted[0]))

# %% significance test
alpha = 0.05
data_A = rmse_valid_GRU[idx_sort_GRU[0], :]
# data_B = rmse_valid_GRU[idx_sort_GRU[3], :]
data_B = rmse_valid_MLP[idx_sort_MLP[0], :]
# data_B = rmse_valid_LSTM[idx_sort_LSTM[0], :]
# data_B = rmse_valid_RNN[idx_sort_RNN[0], :]
t, p = stats.ttest_ind(data_A, data_B, equal_var=False)
if p > alpha:
    print("At the 5% level, fail to reject (accept) the null hypothesis that the means are equal. p={:.5f}".format(p))
else:
    print("At the 5% level, reject the null hypothesis that the means are equal. p={:.5f}".format(p))

# %% get the single best model of each major architecture
# idx_MLP_min = np.argmin(rmse_valid_MLP_min)
# rmse_valid_MLP_min = rmse_valid_MLP_min[idx_MLP_min]
# rmse_train_MLP_min = rmse_train_MLP_min[idx_MLP_min]
# rmse_valid_sdom_MLP_min = rmse_valid_MLP_sdom[idx_MLP_min]
# seq_len_array_MLP_min = seq_len_array[idx_MLP_min]
# num_layers_array_MLP_min = num_layers_array[idx_MLP_min]
# hidden_size_array_MLP_min = hidden_size_array[idx_MLP_min]
# step_size_array_MLP_min = step_size_array[idx_MLP_min]

# print("Minimum best validation dataset loss score is {:.5f}".format(rmse_valid_MLP_min))
# print("Minimum best sequence length: {}".format(seq_len_array_MLP_min))
# print("Minimum best number of layers: {}".format(num_layers_array_MLP_min))
# print("Minimum best number of nodes: {}".format(hidden_size_array_MLP_min))
# print("Minimum best step size: {}".format(step_size_array_MLP_min))


# idx_GRU_min = np.argmin(rmse_valid_GRU_min)
# rmse_valid_GRU_min = rmse_valid_GRU_min[idx_GRU_min]
# rmse_train_GRU_min = rmse_train_GRU_min[idx_GRU_min]
# rmse_valid_sdom_GRU_min = rmse_valid_GRU_sdom[idx_GRU_min]
# seq_len_array_GRU_min = seq_len_array[idx_GRU_min]
# num_layers_array_GRU_min = num_layers_array[idx_GRU_min]
# hidden_size_array_GRU_min = hidden_size_array[idx_GRU_min]
# step_size_array_GRU_min = step_size_array[idx_GRU_min]

# print("Minimum best validation dataset loss score is {:.5f}".format(rmse_valid_GRU_min))
# print("Minimum best sequence length: {}".format(seq_len_array_GRU_min))
# print("Minimum best number of layers: {}".format(num_layers_array_GRU_min))
# print("Minimum best number of nodes: {}".format(hidden_size_array_GRU_min))
# print("Minimum best step size: {}".format(step_size_array_GRU_min))


# idx_RNN_min = np.argmin(rmse_valid_RNN_min)
# rmse_valid_RNN_min = rmse_valid_RNN_min[idx_RNN_min]
# rmse_train_RNN_min = rmse_train_RNN_min[idx_RNN_min]
# rmse_valid_sdom_RNN_min = rmse_valid_RNN_sdom[idx_RNN_min]
# seq_len_array_RNN_min = seq_len_array[idx_RNN_min]
# num_layers_array_RNN_min = num_layers_array[idx_RNN_min]
# hidden_size_array_RNN_min = hidden_size_array[idx_RNN_min]
# step_size_array_RNN_min = step_size_array[idx_RNN_min]

# print("Minimum best validation dataset loss score is {:.5f}".format(rmse_valid_RNN_min))
# print("Minimum best sequence length: {}".format(seq_len_array_RNN_min))
# print("Minimum best number of layers: {}".format(num_layers_array_RNN_min))
# print("Minimum best number of nodes: {}".format(hidden_size_array_RNN_min))
# print("Minimum best step size: {}".format(step_size_array_RNN_min))


# idx_LSTM_min = np.argmin(rmse_valid_LSTM_min)
# rmse_valid_LSTM_min = rmse_valid_LSTM_min[idx_LSTM_min]
# rmse_train_LSTM_min = rmse_train_LSTM_min[idx_LSTM_min]
# rmse_valid_sdom_LSTM_min = rmse_valid_LSTM_sdom[idx_LSTM_min]
# seq_len_array_LSTM_min = seq_len_array[idx_LSTM_min]
# num_layers_array_LSTM_min = num_layers_array[idx_LSTM_min]
# hidden_size_array_LSTM_min = hidden_size_array[idx_LSTM_min]
# step_size_array_LSTM_min = step_size_array[idx_LSTM_min]

# print("Minimum best validation dataset loss score is {:.5f}".format(rmse_valid_LSTM_min))
# print("Minimum best sequence length: {}".format(seq_len_array_LSTM_min))
# print("Minimum best number of layers: {}".format(num_layers_array_LSTM_min))
# print("Minimum best number of nodes: {}".format(hidden_size_array_LSTM_min))
# print("Minimum best step size: {}".format(step_size_array_LSTM_min))

# %%
# idx_MLP_04hr_01lyr = np.logical_and(seq_len_array==4, num_layers_array==1)
# idx_MLP_04hr_02lyr = np.logical_and(seq_len_array==4, num_layers_array==2)
# idx_MLP_04hr_03lyr = np.logical_and(seq_len_array==4, num_layers_array==3)

# idx_MLP_08hr_01lyr = np.logical_and(seq_len_array==8, num_layers_array==1)
# idx_MLP_08hr_02lyr = np.logical_and(seq_len_array==8, num_layers_array==2)
# idx_MLP_08hr_03lyr = np.logical_and(seq_len_array==8, num_layers_array==3)

# idx_MLP_12hr_01lyr = np.logical_and(seq_len_array==12, num_layers_array==1)
# idx_MLP_12hr_02lyr = np.logical_and(seq_len_array==12, num_layers_array==2)
# idx_MLP_12hr_03lyr = np.logical_and(seq_len_array==12, num_layers_array==3)

# idx_MLP_16hr_01lyr = np.logical_and(seq_len_array==16, num_layers_array==1)
# idx_MLP_16hr_02lyr = np.logical_and(seq_len_array==16, num_layers_array==2)
# idx_MLP_16hr_03lyr = np.logical_and(seq_len_array==16, num_layers_array==3)

# =============================================================================
# idx_MLP_04hr_01lyr = np.logical_and(seq_len_array==4,
#                                     np.logical_and(num_layers_array==1,
#                                                    step_size_array==10))
# idx_MLP_04hr_02lyr = np.logical_and(seq_len_array==4,
#                                     np.logical_and(num_layers_array==2,
#                                                    step_size_array==10))
# idx_MLP_04hr_03lyr = np.logical_and(seq_len_array==4,
#                                     np.logical_and(num_layers_array==3,
#                                                    step_size_array==10))
# 
# idx_MLP_08hr_01lyr = np.logical_and(seq_len_array==8,
#                                     np.logical_and(num_layers_array==1,
#                                                    step_size_array==10))
# idx_MLP_08hr_02lyr = np.logical_and(seq_len_array==8,
#                                     np.logical_and(num_layers_array==2,
#                                                    step_size_array==10))
# idx_MLP_08hr_03lyr = np.logical_and(seq_len_array==8,
#                                     np.logical_and(num_layers_array==3,
#                                                    step_size_array==10))
# 
# idx_MLP_12hr_01lyr = np.logical_and(seq_len_array==12,
#                                     np.logical_and(num_layers_array==1,
#                                                    step_size_array==10))
# idx_MLP_12hr_02lyr = np.logical_and(seq_len_array==12,
#                                     np.logical_and(num_layers_array==2,
#                                                    step_size_array==10))
# idx_MLP_12hr_03lyr = np.logical_and(seq_len_array==12,
#                                     np.logical_and(num_layers_array==3,
#                                                    step_size_array==10))
# 
# idx_MLP_16hr_01lyr = np.logical_and(seq_len_array==16,
#                                     np.logical_and(num_layers_array==1,
#                                                    step_size_array==10))
# idx_MLP_16hr_02lyr = np.logical_and(seq_len_array==16,
#                                     np.logical_and(num_layers_array==2,
#                                                    step_size_array==10))
# idx_MLP_16hr_03lyr = np.logical_and(seq_len_array==16,
#                                     np.logical_and(num_layers_array==3,
#                                                    step_size_array==10))
# 
# plt.figure(figsize=(8, 5))
# plt.errorbar(x=hidden_size_array[idx_MLP_04hr_01lyr],
#               y=rmse_train_MLP_mean[idx_MLP_04hr_01lyr],
#               yerr=rmse_train_MLP_sdom[idx_MLP_04hr_01lyr],
#               fmt='-s', color='tab:blue', label='MLP Train: 1 layer')
# plt.errorbar(x=hidden_size_array[idx_MLP_04hr_02lyr],
#               y=rmse_train_MLP_mean[idx_MLP_04hr_02lyr],
#               yerr=rmse_train_MLP_sdom[idx_MLP_04hr_02lyr],
#               fmt='-s', color='tab:orange', label='MLP Train: 2 layers')
# plt.errorbar(x=hidden_size_array[idx_MLP_04hr_03lyr],
#               y=rmse_train_MLP_mean[idx_MLP_04hr_03lyr],
#               yerr=rmse_train_MLP_sdom[idx_MLP_04hr_03lyr],
#               fmt='-s', color='tab:green', label='MLP Train: 3 layers')
# plt.errorbar(x=hidden_size_array[idx_MLP_04hr_01lyr],
#               y=rmse_valid_MLP_mean[idx_MLP_04hr_01lyr],
#               yerr=rmse_valid_MLP_sdom[idx_MLP_04hr_01lyr],
#               fmt='-o', color='tab:blue', label='MLP Valid: 1 layer')
# plt.errorbar(x=hidden_size_array[idx_MLP_04hr_02lyr],
#               y=rmse_valid_MLP_mean[idx_MLP_04hr_02lyr],
#               yerr=rmse_valid_MLP_sdom[idx_MLP_04hr_02lyr],
#               fmt='-o', color='tab:orange', label='MLP Valid: 2 layers')
# plt.errorbar(x=hidden_size_array[idx_MLP_04hr_03lyr],
#               y=rmse_valid_MLP_mean[idx_MLP_04hr_03lyr],
#               yerr=rmse_valid_MLP_sdom[idx_MLP_04hr_03lyr],
#               fmt='-o', color='tab:green', label='MLP Valid: 3 layers')
# plt.ylim(0.15, 0.35)
# plt.title("4hr Sequence Length: Average Loss")
# plt.xlabel('epoch')
# plt.ylabel('$log_{10}(C_{n}^{2})$ RMSE')
# plt.legend(loc='best')
# plt.grid(True)
# plt.grid(True, which='minor')
# plt.tight_layout()
# 
# plt.figure(figsize=(8, 5))
# plt.errorbar(x=hidden_size_array[idx_MLP_08hr_01lyr],
#               y=rmse_train_MLP_mean[idx_MLP_08hr_01lyr],
#               yerr=rmse_train_MLP_sdom[idx_MLP_08hr_01lyr],
#               fmt='-s', color='tab:blue', label='MLP Train: 1 layer')
# plt.errorbar(x=hidden_size_array[idx_MLP_08hr_02lyr],
#               y=rmse_train_MLP_mean[idx_MLP_08hr_02lyr],
#               yerr=rmse_train_MLP_sdom[idx_MLP_08hr_02lyr],
#               fmt='-s', color='tab:orange', label='MLP Train: 2 layers')
# plt.errorbar(x=hidden_size_array[idx_MLP_08hr_03lyr],
#               y=rmse_train_MLP_mean[idx_MLP_08hr_03lyr],
#               yerr=rmse_train_MLP_sdom[idx_MLP_08hr_03lyr],
#               fmt='-s', color='tab:green', label='MLP Train: 3 layers')
# plt.errorbar(x=hidden_size_array[idx_MLP_08hr_01lyr],
#               y=rmse_valid_MLP_mean[idx_MLP_08hr_01lyr],
#               yerr=rmse_valid_MLP_sdom[idx_MLP_08hr_01lyr],
#               fmt='-o', color='tab:blue', label='MLP Valid: 1 layer')
# plt.errorbar(x=hidden_size_array[idx_MLP_08hr_02lyr],
#               y=rmse_valid_MLP_mean[idx_MLP_08hr_02lyr],
#               yerr=rmse_valid_MLP_sdom[idx_MLP_08hr_02lyr],
#               fmt='-o', color='tab:orange', label='MLP Valid: 2 layers')
# plt.errorbar(x=hidden_size_array[idx_MLP_08hr_03lyr],
#               y=rmse_valid_MLP_mean[idx_MLP_08hr_03lyr],
#               yerr=rmse_valid_MLP_sdom[idx_MLP_08hr_03lyr],
#               fmt='-o', color='tab:green', label='MLP Valid: 3 layers')
# plt.ylim(0.15, 0.35)
# plt.title("8hr Sequence Length: Average Loss")
# plt.xlabel('epoch')
# plt.ylabel('$log_{10}(C_{n}^{2})$ RMSE')
# plt.legend(loc='best')
# plt.grid(True)
# plt.grid(True, which='minor')
# plt.tight_layout()
# 
# plt.figure(figsize=(8, 5))
# plt.errorbar(x=hidden_size_array[idx_MLP_12hr_01lyr],
#               y=rmse_train_MLP_mean[idx_MLP_12hr_01lyr],
#               yerr=rmse_train_MLP_sdom[idx_MLP_12hr_01lyr],
#               fmt='-s', color='tab:blue', label='MLP Train: 1 layer')
# plt.errorbar(x=hidden_size_array[idx_MLP_12hr_02lyr],
#               y=rmse_train_MLP_mean[idx_MLP_12hr_02lyr],
#               yerr=rmse_train_MLP_sdom[idx_MLP_12hr_02lyr],
#               fmt='-s', color='tab:orange', label='MLP Train: 2 layers')
# plt.errorbar(x=hidden_size_array[idx_MLP_12hr_03lyr],
#               y=rmse_train_MLP_mean[idx_MLP_12hr_03lyr],
#               yerr=rmse_train_MLP_sdom[idx_MLP_12hr_03lyr],
#               fmt='-s', color='tab:green', label='MLP Train: 3 layers')
# plt.errorbar(x=hidden_size_array[idx_MLP_12hr_01lyr],
#               y=rmse_valid_MLP_mean[idx_MLP_12hr_01lyr],
#               yerr=rmse_valid_MLP_sdom[idx_MLP_12hr_01lyr],
#               fmt='-o', color='tab:blue', label='MLP Valid: 1 layer')
# plt.errorbar(x=hidden_size_array[idx_MLP_12hr_02lyr],
#               y=rmse_valid_MLP_mean[idx_MLP_12hr_02lyr],
#               yerr=rmse_valid_MLP_sdom[idx_MLP_12hr_02lyr],
#               fmt='-o', color='tab:orange', label='MLP Valid: 2 layers')
# plt.errorbar(x=hidden_size_array[idx_MLP_12hr_03lyr],
#               y=rmse_valid_MLP_mean[idx_MLP_12hr_03lyr],
#               yerr=rmse_valid_MLP_sdom[idx_MLP_12hr_03lyr],
#               fmt='-o', color='tab:green', label='MLP Valid: 3 layers')
# plt.ylim(0.15, 0.35)
# plt.title("12hr Sequence Length: Average Loss")
# plt.xlabel('epoch')
# plt.ylabel('$log_{10}(C_{n}^{2})$ RMSE')
# plt.legend(loc='best')
# plt.grid(True)
# plt.grid(True, which='minor')
# plt.tight_layout()
# 
# plt.figure(figsize=(8, 5))
# plt.errorbar(x=hidden_size_array[idx_MLP_16hr_01lyr],
#               y=rmse_train_MLP_mean[idx_MLP_16hr_01lyr],
#               yerr=rmse_train_MLP_sdom[idx_MLP_16hr_01lyr],
#               fmt='-s', color='tab:blue', label='MLP Train: 1 layer')
# plt.errorbar(x=hidden_size_array[idx_MLP_16hr_02lyr],
#               y=rmse_train_MLP_mean[idx_MLP_16hr_02lyr],
#               yerr=rmse_train_MLP_sdom[idx_MLP_16hr_02lyr],
#               fmt='-s', color='tab:orange', label='MLP Train: 2 layers')
# plt.errorbar(x=hidden_size_array[idx_MLP_16hr_03lyr],
#               y=rmse_train_MLP_mean[idx_MLP_16hr_03lyr],
#               yerr=rmse_train_MLP_sdom[idx_MLP_16hr_03lyr],
#               fmt='-s', color='tab:green', label='MLP Train: 3 layers')
# plt.errorbar(x=hidden_size_array[idx_MLP_16hr_01lyr],
#               y=rmse_valid_MLP_mean[idx_MLP_16hr_01lyr],
#               yerr=rmse_valid_MLP_sdom[idx_MLP_16hr_01lyr],
#               fmt='-o', color='tab:blue', label='MLP Valid: 1 layer')
# plt.errorbar(x=hidden_size_array[idx_MLP_16hr_02lyr],
#               y=rmse_valid_MLP_mean[idx_MLP_16hr_02lyr],
#               yerr=rmse_valid_MLP_sdom[idx_MLP_16hr_02lyr],
#               fmt='-o', color='tab:orange', label='MLP Valid: 2 layers')
# plt.errorbar(x=hidden_size_array[idx_MLP_16hr_03lyr],
#               y=rmse_valid_MLP_mean[idx_MLP_16hr_03lyr],
#               yerr=rmse_valid_MLP_sdom[idx_MLP_16hr_03lyr],
#               fmt='-o', color='tab:green', label='MLP Valid: 3 layers')
# plt.ylim(0.15, 0.35)
# plt.title("16hr Sequence Length: Average Loss")
# plt.xlabel('epoch')
# plt.ylabel('$log_{10}(C_{n}^{2})$ RMSE')
# plt.legend(loc='best')
# plt.grid(True)
# plt.grid(True, which='minor')
# plt.tight_layout()
# 
# ###############################################################################
# plt.figure(figsize=(8, 5))
# plt.errorbar(x=hidden_size_array[idx_MLP_04hr_01lyr],
#               y=rmse_train_MLP_min[idx_MLP_04hr_01lyr],
#               yerr=rmse_train_MLP_sdom[idx_MLP_04hr_01lyr],
#               fmt='-s', color='tab:blue', label='MLP Train: 1 layer')
# plt.errorbar(x=hidden_size_array[idx_MLP_04hr_02lyr],
#               y=rmse_train_MLP_min[idx_MLP_04hr_02lyr],
#               yerr=rmse_train_MLP_sdom[idx_MLP_04hr_02lyr],
#               fmt='-s', color='tab:orange', label='MLP Train: 2 layers')
# plt.errorbar(x=hidden_size_array[idx_MLP_04hr_03lyr],
#               y=rmse_train_MLP_min[idx_MLP_04hr_03lyr],
#               yerr=rmse_train_MLP_sdom[idx_MLP_04hr_03lyr],
#               fmt='-s', color='tab:green', label='MLP Train: 3 layers')
# plt.errorbar(x=hidden_size_array[idx_MLP_04hr_01lyr],
#               y=rmse_valid_MLP_min[idx_MLP_04hr_01lyr],
#               yerr=rmse_valid_MLP_sdom[idx_MLP_04hr_01lyr],
#               fmt='-o', color='tab:blue', label='MLP Valid: 1 layer')
# plt.errorbar(x=hidden_size_array[idx_MLP_04hr_02lyr],
#               y=rmse_valid_MLP_min[idx_MLP_04hr_02lyr],
#               yerr=rmse_valid_MLP_sdom[idx_MLP_04hr_02lyr],
#               fmt='-o', color='tab:orange', label='MLP Valid: 2 layers')
# plt.errorbar(x=hidden_size_array[idx_MLP_04hr_03lyr],
#               y=rmse_valid_MLP_min[idx_MLP_04hr_03lyr],
#               yerr=rmse_valid_MLP_sdom[idx_MLP_04hr_03lyr],
#               fmt='-o', color='tab:green', label='MLP Valid: 3 layers')
# plt.ylim(0.15, 0.35)
# plt.title("4hr Sequence Length: Minimum Loss")
# plt.xlabel('epoch')
# plt.ylabel('$log_{10}(C_{n}^{2})$ RMSE')
# plt.legend(loc='best')
# plt.grid(True)
# plt.grid(True, which='minor')
# plt.tight_layout()
# 
# plt.figure(figsize=(8, 5))
# plt.errorbar(x=hidden_size_array[idx_MLP_08hr_01lyr],
#               y=rmse_train_MLP_min[idx_MLP_08hr_01lyr],
#               yerr=rmse_train_MLP_sdom[idx_MLP_08hr_01lyr],
#               fmt='-s', color='tab:blue', label='MLP Train: 1 layer')
# plt.errorbar(x=hidden_size_array[idx_MLP_08hr_02lyr],
#               y=rmse_train_MLP_min[idx_MLP_08hr_02lyr],
#               yerr=rmse_train_MLP_sdom[idx_MLP_08hr_02lyr],
#               fmt='-s', color='tab:orange', label='MLP Train: 2 layers')
# plt.errorbar(x=hidden_size_array[idx_MLP_08hr_03lyr],
#               y=rmse_train_MLP_min[idx_MLP_08hr_03lyr],
#               yerr=rmse_train_MLP_sdom[idx_MLP_08hr_03lyr],
#               fmt='-s', color='tab:green', label='MLP Train: 3 layers')
# plt.errorbar(x=hidden_size_array[idx_MLP_08hr_01lyr],
#               y=rmse_valid_MLP_min[idx_MLP_08hr_01lyr],
#               yerr=rmse_valid_MLP_sdom[idx_MLP_08hr_01lyr],
#               fmt='-o', color='tab:blue', label='MLP Valid: 1 layer')
# plt.errorbar(x=hidden_size_array[idx_MLP_08hr_02lyr],
#               y=rmse_valid_MLP_min[idx_MLP_08hr_02lyr],
#               yerr=rmse_valid_MLP_sdom[idx_MLP_08hr_02lyr],
#               fmt='-o', color='tab:orange', label='MLP Valid: 2 layers')
# plt.errorbar(x=hidden_size_array[idx_MLP_08hr_03lyr],
#               y=rmse_valid_MLP_min[idx_MLP_08hr_03lyr],
#               yerr=rmse_valid_MLP_sdom[idx_MLP_08hr_03lyr],
#               fmt='-o', color='tab:green', label='MLP Valid: 3 layers')
# plt.ylim(0.15, 0.35)
# plt.title("8hr Sequence Length: Minimum Loss")
# plt.xlabel('epoch')
# plt.ylabel('$log_{10}(C_{n}^{2})$ RMSE')
# plt.legend(loc='best')
# plt.grid(True)
# plt.grid(True, which='minor')
# plt.tight_layout()
# 
# plt.figure(figsize=(8, 5))
# plt.errorbar(x=hidden_size_array[idx_MLP_12hr_01lyr],
#               y=rmse_train_MLP_min[idx_MLP_12hr_01lyr],
#               yerr=rmse_train_MLP_sdom[idx_MLP_12hr_01lyr],
#               fmt='-s', color='tab:blue', label='MLP Train: 1 layer')
# plt.errorbar(x=hidden_size_array[idx_MLP_12hr_02lyr],
#               y=rmse_train_MLP_min[idx_MLP_12hr_02lyr],
#               yerr=rmse_train_MLP_sdom[idx_MLP_12hr_02lyr],
#               fmt='-s', color='tab:orange', label='MLP Train: 2 layers')
# plt.errorbar(x=hidden_size_array[idx_MLP_12hr_03lyr],
#               y=rmse_train_MLP_min[idx_MLP_12hr_03lyr],
#               yerr=rmse_train_MLP_sdom[idx_MLP_12hr_03lyr],
#               fmt='-s', color='tab:green', label='MLP Train: 3 layers')
# plt.errorbar(x=hidden_size_array[idx_MLP_12hr_01lyr],
#               y=rmse_valid_MLP_min[idx_MLP_12hr_01lyr],
#               yerr=rmse_valid_MLP_sdom[idx_MLP_12hr_01lyr],
#               fmt='-o', color='tab:blue', label='MLP Valid: 1 layer')
# plt.errorbar(x=hidden_size_array[idx_MLP_12hr_02lyr],
#               y=rmse_valid_MLP_min[idx_MLP_12hr_02lyr],
#               yerr=rmse_valid_MLP_sdom[idx_MLP_12hr_02lyr],
#               fmt='-o', color='tab:orange', label='MLP Valid: 2 layers')
# plt.errorbar(x=hidden_size_array[idx_MLP_12hr_03lyr],
#               y=rmse_valid_MLP_min[idx_MLP_12hr_03lyr],
#               yerr=rmse_valid_MLP_sdom[idx_MLP_12hr_03lyr],
#               fmt='-o', color='tab:green', label='MLP Valid: 3 layers')
# plt.ylim(0.15, 0.35)
# plt.title("12hr Sequence Length: Minimum Loss")
# plt.xlabel('epoch')
# plt.ylabel('$log_{10}(C_{n}^{2})$ RMSE')
# plt.legend(loc='best')
# plt.grid(True)
# plt.grid(True, which='minor')
# plt.tight_layout()
# 
# plt.figure(figsize=(8, 5))
# plt.errorbar(x=hidden_size_array[idx_MLP_16hr_01lyr],
#               y=rmse_train_MLP_min[idx_MLP_16hr_01lyr],
#               yerr=rmse_train_MLP_sdom[idx_MLP_16hr_01lyr],
#               fmt='-s', color='tab:blue', label='MLP Train: 1 layer')
# plt.errorbar(x=hidden_size_array[idx_MLP_16hr_02lyr],
#               y=rmse_train_MLP_min[idx_MLP_16hr_02lyr],
#               yerr=rmse_train_MLP_sdom[idx_MLP_16hr_02lyr],
#               fmt='-s', color='tab:orange', label='MLP Train: 2 layers')
# plt.errorbar(x=hidden_size_array[idx_MLP_16hr_03lyr],
#               y=rmse_train_MLP_min[idx_MLP_16hr_03lyr],
#               yerr=rmse_train_MLP_sdom[idx_MLP_16hr_03lyr],
#               fmt='-s', color='tab:green', label='MLP Train: 3 layers')
# plt.errorbar(x=hidden_size_array[idx_MLP_16hr_01lyr],
#               y=rmse_valid_MLP_min[idx_MLP_16hr_01lyr],
#               yerr=rmse_valid_MLP_sdom[idx_MLP_16hr_01lyr],
#               fmt='-o', color='tab:blue', label='MLP Valid: 1 layer')
# plt.errorbar(x=hidden_size_array[idx_MLP_16hr_02lyr],
#               y=rmse_valid_MLP_min[idx_MLP_16hr_02lyr],
#               yerr=rmse_valid_MLP_sdom[idx_MLP_16hr_02lyr],
#               fmt='-o', color='tab:orange', label='MLP Valid: 2 layers')
# plt.errorbar(x=hidden_size_array[idx_MLP_16hr_03lyr],
#               y=rmse_valid_MLP_min[idx_MLP_16hr_03lyr],
#               yerr=rmse_valid_MLP_sdom[idx_MLP_16hr_03lyr],
#               fmt='-o', color='tab:green', label='MLP Valid: 3 layers')
# plt.ylim(0.15, 0.35)
# plt.title("16hr Sequence Length: Minimum Loss")
# plt.xlabel('epoch')
# plt.ylabel('$log_{10}(C_{n}^{2})$ RMSE')
# plt.legend(loc='best')
# plt.grid(True)
# plt.grid(True, which='minor')
# plt.tight_layout()
# 
# =============================================================================

# # %% sort validation dataset results
# idx_sort = np.argsort(rmse_valid_GRU_mean)
# rmse_valid_GRU_mean_sorted = rmse_valid_GRU_mean[idx_sort]
# rmse_valid_GRU_sdom_sorted = rmse_valid_GRU_sdom[idx_sort]
# seq_len_array_sorted = seq_len_array[idx_sort]
# num_layers_array_sorted = num_layers_array[idx_sort]
# hidden_size_array_sorted = hidden_size_array[idx_sort]

# # %% significance test
# alpha = 0.05
# data_A = rmse_valid_GRU[idx_sort[0],:]
# data_B = rmse_valid_GRU[idx_sort[9],:]
# t, p = stats.ttest_ind(data_A, data_B, equal_var=False)
# if p > alpha:
#     print("At the 5% level, fail to reject (accept) the null hypothesis that the means are equal.")
# else:
#     print("At the 5% level, reject the null hypothesis that the means are equal.")

