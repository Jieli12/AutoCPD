"""
Author         : Jie Li, Department of Statistics, London School of Economics.
Date           : 2022-11-04 22:57:25
Last Revision  : 2022-11-06 09:48:47
Last Author    : Jie Li
File Path      : /AutoCPD/Code/ScenarioS1Rho0L10Predict.py
Description    :








Copyright (c) 2022 by Jie Li, j.li196@lse.ac.uk
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
import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import tensorflow_docs.plots
from keras import layers, losses, metrics, models
# %%
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from DataSetGen import *
from utils import *

# %% parameter settings
n_vec = np.array([100], dtype=np.int32)  # the length of time series
n_len = len(n_vec)
epsilon = 0.05
N_test = 500  # the sample size
N_vec = np.arange(100, 800, 100, dtype=np.int32)  # the sample size
B = np.sqrt(8 * np.log(n_vec / epsilon) / n_vec)
mu_L = 0
tau_bound = 2
num_repeat = 30
B_bound = np.array([0.25, 1.75])
rho = 0.0
#  setup the tensorboard
file_path = Path(__file__)
cusum_result_folder = Path(file_path.parents[1], "datasets", "CusumResult")
current_file = file_path.stem
current_file = current_file.replace('Predict', 'Train')
print(current_file)
logdir = Path("tensorboard_logs", "Trial")

# load the cusum thresholds
# pkl_path = Path(cusum_result_folder, current_file + '.pkl')
# Cusum_th = pd.read_pickle(pkl_path)
num_models = 7
# %% prediction
n = n_vec[0]
N = int(N_test * num_repeat / 2)
result_nnL10 = np.empty((num_models, num_repeat, 5))
for i in range(num_models):
	# n = Cusum_th.at[i, 'n']
	N_train = N_vec[i]
	# B = Cusum_th.at[i, 'B']
	# threshold_opt = Cusum_th.at[i, 'Threshold']
	# print(n, B, threshold_opt)

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
		type='Gaussian'
	)
	data_alt = result["data"]
	#  generate dataset for null hypothesis
	data_null = GenDataMean(N, n, cp=None, mu=(mu_L, mu_L), sigma=1)
	data_all = np.concatenate((data_alt, data_null), axis=0)
	y_all = np.repeat((1, 0), N).reshape((2 * N, 1))
	#  generate the training dataset and test dataset
	data_all, y_all = shuffle(data_all, y_all, random_state=42)

	# CUSUM prediction
	# y_cusum_test_max = np.apply_along_axis(MaxCUSUM, 1, data_all)
	# y_pred_cusum_all = y_cusum_test_max > threshold_opt

	# load NN Classifiers
	Q = int(np.floor(np.log2(n / 2))) + 1
	# the number of hidden nodes
	m = 4 * Q
	model_name = current_file + "n" + str(n) + "N" + str(N_train) + "m" + str(m)
	model_path = Path(logdir, model_name, 'model')
	# model_path_3.append(model_path)

	modelL10 = tf.keras.models.load_model(model_path)
	y_pred_NN_allL10 = np.argmax(modelL10.predict(data_all), axis=1)

	for j in range(num_repeat):
		# cusum
		ind = range(N_test * j, N_test * (j + 1))
		y_test = y_all[ind, 0]
		# print("MER of CUSUM:", np.array(mer_cusum))
		# NN0
		y_pred_nnL10 = y_pred_NN_allL10[ind]
		confusion_mtx = tf.math.confusion_matrix(y_test, y_pred_nnL10)
		mer_nnL10 = (confusion_mtx[0, 1] + confusion_mtx[1, 0]) / N_test
		result_nnL10[i, j, 0] = mer_nnL10
		result_nnL10[i, j, 1:] = np.reshape(confusion_mtx, (4,))

# %%
# save the cusum threshold to folder datasets/CusumResult/
nnL10_vec = np.mean(result_nnL10, axis=1, keepdims=False)[:, 0]
path_nnL10 = Path(cusum_result_folder, current_file + "result_nnL10")
np.save(path_nnL10, result_nnL10)

# %%
