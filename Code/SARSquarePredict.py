"""
Author         : Jie Li, Department of Statistics, London School of Economics.
Date           : 2023-05-11 09:45:48
Last Revision  : 2023-09-04 14:55:49
Last Author    : Jie Li
File Path      : /AutoCPD/Code/SARSquarePredict.py
Description    :


Figure R2(b), predict, double checked





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
n_vec = np.array([200], dtype=np.int32)  # the length of time series
n_len = len(n_vec)
Q = np.array(np.floor(np.log2(n_vec / 2)), dtype=np.int32) + 1
# the number of hidden nodes
m_mat = np.c_[4 * Q, 4 * Q, 2 * n_vec - 2]
L_mat = np.c_[1, 5, 1]
N_vec = np.arange(100, 800, 100, dtype=np.int32)  # the sample size
num_N = len(N_vec)
coef_left = 0.2
coef_right = 0.8
sigma = 0.25
tau_bound = 10
#  setup the tensorboard
file_path = Path(__file__)
result_folder = Path(file_path.parents[1], "datasets", "CusumResult")
current_file = file_path.stem
current_file = current_file.replace('Predict', 'Train')
print(current_file)
logdir = Path("tensorboard_logs", "Trial")

N_test = 500
num_repeat = 30
# %% prediction
N = N_test * num_repeat
result_nn = np.empty((7, 3, num_repeat, 5))
for i in range(7):
	n = n_vec[0]
	N_train = N_vec[i]
	np.random.seed(2022)  # numpy seed fixing
	tf.random.set_seed(2022)  # tensorflow seed fixing
	data_all, tau_alt = GenerateARAll(N, int(n/2), coef_left, coef_right, sigma, tau_bound)
	data_all = np.concatenate(
		[data_all, data_all[:, :99] * data_all[:, 1:]], axis=1
	)
	n = data_all.shape[1]
	#  generate dataset for null hypothesis
	y_all = np.repeat((0, 1), N).reshape((2 * N, 1))
	#  generate the training dataset and test dataset
	data_all, y_all = shuffle(data_all, y_all, random_state=42)

	for k in range(3):
		m = m_mat[0, k]
		l = L_mat[0, k]
		model_name = current_file + "n" + str(n) + "N" + str(
			N_train
		) + "m" + str(m) + "L" + str(l)
		model_path = Path(logdir, model_name, 'model')
		modelnn = tf.keras.models.load_model(model_path)
		y_pred_NN_all = np.argmax(modelnn.predict(data_all), axis=1)

		for j in range(num_repeat):
			# cusum
			ind = range(2 * N_test * j, 2 * N_test * (j + 1))
			y_test = y_all[ind, 0]
			# print("MER of CUSUM:", np.array(mer_cusum))
			# NN0
			y_pred_nn = y_pred_NN_all[ind]
			confusion_mtx = tf.math.confusion_matrix(y_test, y_pred_nn)
			mer_nn = (confusion_mtx[0, 1] + confusion_mtx[1, 0]) / N_test / 2
			result_nn[i, k, j, 0] = mer_nn
			result_nn[i, k, j, 1:] = np.reshape(confusion_mtx, (4,))

# %%
path_nn = Path(result_folder, current_file + "result_nnARsquare")
np.save(path_nn, result_nn)

# %%
