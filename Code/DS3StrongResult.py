"""
Author         : Jie Li, Department of Statistics, London School of Economics.
Date           : 2023-05-04 15:41:44
Last Revision  : 2023-09-04 16:30:18
Last Author    : Jie Li
File Path      : /AutoCPD/Code/DS3StrongResult.py
Description    :


Detection, strong SNR, n=300,400,500,600; n_prime=2000, tau_bound=750, N_prime=500
S3(heavy tail) with B_bound=[1, 3]
Compute Result





Copyright (c) 2023 by Jie Li, j.li196@lse.ac.uk
All Rights Reserved.
"""

# %%
import pathlib
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from DataSetGen import *
from keras import layers, losses, metrics, models
from matplotlib.ticker import StrMethodFormatter
from sklearn.metrics import auc, mean_squared_error, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from utils import *

# %% parameter settings
print("num:", len(tf.config.list_physical_devices('GPU')))
n_candidates = np.array(
	[300, 400, 500, 600], dtype=np.int32
)  # the length of time series
epsilon = 0.05
n_prime = 2000  # the sample size
B = np.sqrt(8 * np.log(n_prime / epsilon) / n_prime)
mu_L = 0
tau_bound = 750
B_bound = np.array([1, 3])
N_prime = 500
# parameters for neural network
learning_rate = 1e-3
epochs = 300
batch_size = 32
num_classes = 2
sample_size = 30
ntrim = 10
L = 1
m_candidate = 2 * n_candidates - 2
N_test = 3000
#  setup the tensorboard
file_path = Path(__file__)
cusum_result_folder = Path(file_path.parents[1], "datasets", "CusumResult")
current_file = file_path.stem
current_file = current_file.replace('Result', '')
print(current_file)
logdir = Path("tensorboard_logs", "Trial")
np.random.seed(2023)  # numpy seed fixing
tf.random.set_seed(2023)  # tensorflow seed fixing

rmse = np.zeros((len(n_candidates), 5))
for j, n, m in zip(range(len(n_candidates)), n_candidates, m_candidate):
	# print([n,m])
	model_name = current_file + "n" + str(n)
	print(model_name)
	model_path = Path(logdir, model_name, 'model')
	model = tf.keras.models.load_model(model_path)
	# generate time series with length N_prime
	result = DataGenAlternative(
		N_test,
		B,
		mu_L,
		n_prime,
		tau_bound=tau_bound,
		B_bound=B_bound,
		type='ARH',
		scale=0.005
	)
	data = result["data"]
	tau_all = result["tau_alt"]

	cusum_loc = np.apply_along_axis(get_cusum_location, 1, data)
	mosum_loc_LR = np.apply_along_axis(ComputeMosum, 1, data, n)
	mosum_loc_3 = np.zeros((N_test, 3))
	for i in range(N_test):
		mosum_loc_3[i, :] = get_loc_3(model, data[i, :], n, n)
	rmse_mosum_loc_3 = np.sqrt(
		np.mean((mosum_loc_3 - np.reshape(tau_all, (N_test, 1)))**2, axis=0)
	)
	rmse_cusum = np.sqrt(mean_squared_error(tau_all, cusum_loc))
	rmse_mosum_LR = np.sqrt(mean_squared_error(tau_all, mosum_loc_LR))
	rmse_tmp = np.concatenate(
		(np.array([rmse_cusum, rmse_mosum_LR]), rmse_mosum_loc_3)
	)
	rmse[j, :] = rmse_tmp
np.save("../datasets/CusumResult/DS3Strong", rmse)
np.save("DS3Strong", rmse)
np.round(rmse, 2)
