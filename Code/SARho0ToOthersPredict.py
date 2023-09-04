"""
Author         : Jie Li, Department of Statistics, London School of Economics.
Date           : 2023-03-27 09:23:27
Last Revision  : 2023-09-04 14:29:48
Last Author    : Jie Li
File Path      : /AutoCPD/Code/SARho0ToOthersPredict.py
Description    : Appling Scenario 1 with rho=0 to other 3 Scenarios and plot the figures together.


Figure R3, double checked





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
import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import tensorflow_docs.plots
from DataSetGen import *
from keras import layers, losses, metrics, models
from matplotlib.ticker import StrMethodFormatter
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
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
current_file = current_file.replace(
	'SARho0ToOthersPredict', 'ScenarioARho0Train'
)
print(current_file)
logdir = Path("tensorboard_logs", "Trial")

# load the cusum thresholds
pkl_path = Path(cusum_result_folder, current_file + '.pkl')
Cusum_th = pd.read_pickle(pkl_path)
num_models = Cusum_th.shape[0]
# %% prediction
N = int(N_test * num_repeat / 2)
result_cusum = np.empty((num_models, 4, num_repeat, 5))
result_nn1 = np.empty((num_models, 4, num_repeat, 5))
result_nn2 = np.empty((num_models, 4, num_repeat, 5))
result_nnL5 = np.empty((num_models, 4, num_repeat, 5))
result_nnL10 = np.empty((num_models, 4, num_repeat, 5))
Scenarios = ["A0", "A07", "C", "D"]
for i in range(num_models):
	n = Cusum_th.at[i, 'n']
	N_train = Cusum_th.at[i, 'N']
	B = Cusum_th.at[i, 'B']
	threshold_opt = Cusum_th.at[i, 'Threshold']
	print(n, B, threshold_opt)

	#  generate the dataset
	for count, scenario in enumerate(Scenarios):
		data_all, y_all = DataGenScenarios(scenario, N, B, mu_L, n, rho, tau_bound, B_bound)

		# CUSUM prediction
		y_cusum_test_max = np.apply_along_axis(MaxCUSUM, 1, data_all)
		y_pred_cusum_all = y_cusum_test_max > threshold_opt

		# load NN Classifiers
		Q = int(np.floor(np.log2(n / 2))) + 1
		# the number of hidden nodes
		m_vec = np.array([4 * Q, 2 * n - 2])
		model_path_4 = []
		for k in range(2):
			m = m_vec[k]
			suffix = "n" + str(n) + "N" + str(N_train) + "m" + str(m)
			model_name = current_file + suffix
			model_path = Path(logdir, model_name, 'model')
			model_path_4.append(model_path)
		suffix1 = "n" + str(n) + "N" + str(N_train) + "m" + str(4 * Q)
		model_name = current_file.replace('Train', 'L5Train') + suffix1
		model_path = Path(logdir, model_name, 'model')
		model_path_4.append(model_path)
		model_name = current_file.replace('Train', 'L10Train') + suffix1
		model_path = Path(logdir, model_name, 'model')
		model_path_4.append(model_path)

		# load models
		model1 = tf.keras.models.load_model(model_path_4[0])
		model2 = tf.keras.models.load_model(model_path_4[1])
		modelL5 = tf.keras.models.load_model(model_path_4[2])
		modelL10 = tf.keras.models.load_model(model_path_4[3])
		y_pred_NN_all1 = np.argmax(model1.predict(data_all), axis=1)
		y_pred_NN_all2 = np.argmax(model2.predict(data_all), axis=1)
		y_pred_NN_allL5 = np.argmax(modelL5.predict(data_all), axis=1)
		y_pred_NN_allL10 = np.argmax(modelL10.predict(data_all), axis=1)

		for j in range(num_repeat):
			# cusum
			ind = range(N_test * j, N_test * (j + 1))
			y_test = y_all[ind, 0]
			y_pred_cusum = y_pred_cusum_all[ind]
			confusion_mtx = tf.math.confusion_matrix(y_test, y_pred_cusum)
			mer_cusum = (confusion_mtx[0, 1] + confusion_mtx[1, 0]) / N_test
			result_cusum[i, count, j, 0] = mer_cusum
			result_cusum[i, count, j, 1:] = np.reshape(confusion_mtx, (4,))
			# print("MER of CUSUM:", np.array(mer_cusum))

			# NN1
			y_pred_nn1 = y_pred_NN_all1[ind]
			confusion_mtx = tf.math.confusion_matrix(y_test, y_pred_nn1)
			mer_nn1 = (confusion_mtx[0, 1] + confusion_mtx[1, 0]) / N_test
			result_nn1[i, count, j, 0] = mer_nn1
			result_nn1[i, count, j, 1:] = np.reshape(confusion_mtx, (4,))
			# print("MER of NN1:", np.array(mer_nn1))

			# NN2
			y_pred_nn2 = y_pred_NN_all2[ind]
			confusion_mtx = tf.math.confusion_matrix(y_test, y_pred_nn2)
			mer_nn2 = (confusion_mtx[0, 1] + confusion_mtx[1, 0]) / N_test
			result_nn2[i, count, j, 0] = mer_nn2
			result_nn2[i, count, j, 1:] = np.reshape(confusion_mtx, (4,))
			# print("MER of NN2:", np.array(mer_nn2))

			# NNL5
			y_pred_nnL5 = y_pred_NN_allL5[ind]
			confusion_mtx = tf.math.confusion_matrix(y_test, y_pred_nnL5)
			mer_nnL5 = (confusion_mtx[0, 1] + confusion_mtx[1, 0]) / N_test
			result_nnL5[i, count, j, 0] = mer_nnL5
			result_nnL5[i, count, j, 1:] = np.reshape(confusion_mtx, (4,))
			# print("MER of NN2:", np.array(mer_nnL5))

			# NNL10
			y_pred_nnL10 = y_pred_NN_allL10[ind]
			confusion_mtx = tf.math.confusion_matrix(y_test, y_pred_nnL10)
			mer_nnL10 = (confusion_mtx[0, 1] + confusion_mtx[1, 0]) / N_test
			result_nnL10[i, count, j, 0] = mer_nnL10
			result_nnL10[i, count, j, 1:] = np.reshape(confusion_mtx, (4,))
			# print("MER of NN2:", np.array(mer_nnL10))

# %%
# save the cusum threshold to folder datasets/CusumResult/
path_cusum = Path(cusum_result_folder, current_file + "A2OtherResult_cusum")
path_nn1 = Path(cusum_result_folder, current_file + "A2OtherResult_nn1")
path_nn2 = Path(cusum_result_folder, current_file + "A2OtherResult_nn2")
path_nnL5 = Path(cusum_result_folder, current_file + "A2OtherResult_nnL5")
path_nnL10 = Path(cusum_result_folder, current_file + "A2OtherResult_nnL10")
np.save(path_cusum, result_cusum)
np.save(path_nn1, result_nn1)
np.save(path_nn2, result_nn2)
np.save(path_nnL5, result_nnL5)
np.save(path_nnL10, result_nnL10)
# %%
