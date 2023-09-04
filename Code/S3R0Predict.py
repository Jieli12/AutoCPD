"""
Author         : Jie Li, Department of Statistics, London School of Economics.
Date           : 2023-04-12 07:58:01
Last Revision  : 2023-09-04 13:14:51
Last Author    : Jie Li
File Path      : /AutoCPD/Code/S3R0Predict.py
Description    : Scenario 3, rho=0, hidden layer=1, number of threshold values=6



double checked, Fig.S1 in Supplement




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
# %%
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from utils import *

# %% parameter settings
n = 100  # the length of time series
epsilon = 0.05
N_test = 500  # the sample size
N_vec = np.arange(100, 1100, 100, dtype=np.int32)  # the sample size
num_N = len(N_vec)
m = 2 * n - 2
B = np.sqrt(8 * np.log(n / epsilon) / n)
mu_L = 0
tau_bound = 2
num_repeat = 30
B_bound = np.array([0.25, 1.75])
rho = 0.0
scale = 0.3
#  setup the tensorboard
file_path = Path(__file__)
result_folder = Path(file_path.parents[1], "datasets", "CusumResult")
current_file = file_path.stem
current_file = current_file.replace('Predict', 'Train')
print(current_file)
logdir = Path("tensorboard_logs", "Trial")

# load the cusum thresholds
pkl_path = Path(result_folder, current_file + 'cusum.pkl')
Cusum_th = pd.read_pickle(pkl_path)
pkl_path = Path(result_folder, current_file + 'wilcox.pkl')
Wilcox_th = pd.read_pickle(pkl_path)
num_models = num_N
truncate_value = np.arange(2, 4, 0.5)


def get_truncated(data, thresh):
	x_mean = np.mean(data)
	x_std = np.std(data)
	c = thresh * x_std
	c1 = np.median(data)
	data[data > x_mean + c] = c + x_mean
	data[data < x_mean - c] = x_mean - c
	return data


# %% prediction
N = int(N_test * num_repeat / 2)
result_cusum = np.empty((num_models, num_repeat, 5))
result_wilcox = np.empty((num_models, num_repeat, 5))
result_nn = np.empty((num_models, len(truncate_value) + 1, num_repeat, 5))

for i in range(num_models):
	n = Cusum_th.at[i, 'n']
	N_train = Cusum_th.at[i, 'N']
	B = Cusum_th.at[i, 'B']
	th_opt_cusum = Cusum_th.at[i, 'Threshold']
	th_opt_wilcox = Wilcox_th.at[i, 'Threshold']
	print(n, B, th_opt_cusum, th_opt_wilcox)
	#  generate the dataset for alternative hypothesis
	np.random.seed(2022)  # numpy seed fixing
	tf.random.set_seed(2022)  # tensorflow seed fixing
	result = DataGenAlternative(
		N_sub=N,
		B=B,
		mu_L=mu_L,
		n=n,
		ARcoef=rho,
		tau_bound=tau_bound,
		B_bound=B_bound,
		type='ARH',
		scale=scale
	)
	data_alt = result["data"]
	#  generate dataset for null hypothesis
	data_null = GenDataMeanARH(
		N, n, cp=None, mu=(mu_L, mu_L), coef=rho, scale=scale
	)
	data_all = np.concatenate((data_alt, data_null), axis=0)
	y_all = np.repeat((1, 0), N).reshape((2 * N, 1))
	#  generate the training dataset and test dataset
	data_all, y_all = shuffle(data_all, y_all, random_state=42)

	# NN no truncation
	model_name = current_file + "N" + str(N_train) + "m" + str(m)
	print(model_name)
	model_path = Path(logdir, model_name, 'model')
	model0 = tf.keras.models.load_model(model_path)
	y_pred_NN0_all = np.argmax(model0.predict(data_all), axis=1)

	# CUSUM prediction
	y_cusum_test_max = np.apply_along_axis(MaxCUSUM, 1, data_all)
	y_pred_cusum_all = y_cusum_test_max > th_opt_cusum
	# Wilcox prediction
	y_wilcox_test_max = np.apply_along_axis(get_wilcoxon_test, 1, data_all)
	y_pred_wilcox_all = y_wilcox_test_max > th_opt_wilcox

	for j in range(num_repeat):
		# cusum
		ind = range(N_test * j, N_test * (j + 1))
		y_test = y_all[ind, 0]
		y_pred_cusum = y_pred_cusum_all[ind]
		confusion_mtx = tf.math.confusion_matrix(y_test, y_pred_cusum)
		mer_cusum = (confusion_mtx[0, 1] + confusion_mtx[1, 0]) / N_test
		result_cusum[i, j, 0] = mer_cusum
		result_cusum[i, j, 1:] = np.reshape(confusion_mtx, (4,))
		# wilcox
		y_pred_wilcox = y_pred_wilcox_all[ind]
		confusion_mtx = tf.math.confusion_matrix(y_test, y_pred_wilcox)
		mer_wilcox = (confusion_mtx[0, 1] + confusion_mtx[1, 0]) / N_test
		result_wilcox[i, j, 0] = mer_wilcox
		result_wilcox[i, j, 1:] = np.reshape(confusion_mtx, (4,))
		# nn0
		y_pred_nn0 = y_pred_NN0_all[ind]
		confusion_mtx = tf.math.confusion_matrix(y_test, y_pred_nn0)
		mer_nn0 = (confusion_mtx[0, 1] + confusion_mtx[1, 0]) / N_test
		result_nn[i, 0, j, 0] = mer_nn0
		result_nn[i, 0, j, 1:] = np.reshape(confusion_mtx, (4,))
		# print("MER of CUSUM:", np.array(mer_wilcox))
	# load NN Classifiers

	for t in range(len(truncate_value)):
		data_allT = np.apply_along_axis(
			get_truncated, 1, data_all, truncate_value[t]
		)
		model_name = current_file + "N" + str(N_train
												) + "m" + str(m) + "T" + str(
													truncate_value[t]
												)
		print(model_name)
		model_path = Path(logdir, model_name, 'model')
		model0 = tf.keras.models.load_model(model_path)
		y_pred_NN_all = np.argmax(model0.predict(data_allT), axis=1)
		for j in range(num_repeat):
			# cusum
			ind = range(N_test * j, N_test * (j + 1))
			y_test = y_all[ind, 0]
			y_pred_nn0 = y_pred_NN_all[ind]
			confusion_mtx = tf.math.confusion_matrix(y_test, y_pred_nn0)
			mer_nn0 = (confusion_mtx[0, 1] + confusion_mtx[1, 0]) / N_test
			result_nn[i, t + 1, j, 0] = mer_nn0
			result_nn[i, t + 1, j, 1:] = np.reshape(confusion_mtx, (4,))

# %%

path_cusum = Path(result_folder, current_file + "result_cusumT")
path_wilcox = Path(result_folder, current_file + "result_wilcoxT")
path_nn = Path(result_folder, current_file + "result_nnT")
np.save(path_cusum, result_cusum)
np.save(path_wilcox, result_wilcox)
np.save(path_nn, result_nn)

# %%
