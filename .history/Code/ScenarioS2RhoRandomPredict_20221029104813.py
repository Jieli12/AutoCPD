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

from utils import *

# %% parameter settings
n_vec = np.array([100], dtype=np.int32)  # the length of time series
n_len = len(n_vec)
epsilon = 0.05
N_test = 500  # the sample size
N_vec = np.arange(100, 1100, 100, dtype=np.int32)  # the sample size
B = np.sqrt(8 * np.log(n_vec / epsilon) / n_vec)
mu_L = 0
tau_bound = 2
num_repeat = 30
sigma = np.sqrt(2)
B_bound = np.array([0.25, 1.75])
#  setup the tensorboard
file_path = Path(__file__)
cusum_result_folder = Path(file_path.parents[1], "datasets", "CusumResult")
current_file = file_path.stem
current_file = current_file.replace('Predict', 'Train')
print(current_file)
logdir = Path("tensorboard_logs", "Trial")

# load the cusum thresholds
pkl_path = Path(cusum_result_folder, current_file + '.pkl')
Cusum_th = pd.read_pickle(pkl_path)
num_models = Cusum_th.shape[0]
# %% prediction
N = int(N_test * num_repeat / 2)
result_cusum = np.empty((num_models, num_repeat, 5))
result_nn0 = np.empty((num_models, num_repeat, 5))
result_nn1 = np.empty((num_models, num_repeat, 5))
result_nn2 = np.empty((num_models, num_repeat, 5))
for i in range(num_models):
	n = Cusum_th.at[i, 'n']
	N_train = Cusum_th.at[i, 'N']
	B = Cusum_th.at[i, 'B']
	threshold_opt = Cusum_th.at[i, 'Threshold']
	print(n, B, threshold_opt)

	#  generate the dataset for alternative hypothesis
	np.random.seed(2022)  # numpy seed fixing
	tf.random.set_seed(2022)  # tensorflow seed fixing
	result = DataGenAlternative(
		N_sub=N,
		B=B,
		mu_L=mu_L,
		n=n,
		tau_bound=tau_bound,
		B_bound=B_bound,
		type='ARrho',
		sigma=sigma,
	)
	data_alt = result["data"]
	#  generate dataset for null hypothesis
	data_null = GenDataMeanARrho(N, n, cp=None, mu=(mu_L, mu_L), sigma=sigma)
	data_all = np.concatenate((data_alt, data_null), axis=0)
	y_all = np.repeat((1, 0), N).reshape((2 * N, 1))
	#  generate the training dataset and test dataset
	data_all, y_all = shuffle(data_all, y_all, random_state=42)

	# CUSUM prediction
	y_cusum_test_max = np.apply_along_axis(MaxCUSUM, 1, data_all)
	y_pred_cusum_all = y_cusum_test_max > threshold_opt

	# load NN Classifiers
	Q = int(np.floor(np.log2(n / 2))) + 1
	# the number of hidden nodes
	m_vec = np.array([3, 4 * Q, 2 * n - 2])
	model_path_3 = []
	for k in range(3):
		m = m_vec[k]
		model_name = current_file + "n" + str(n) + "N" + str(N_train
															) + "m" + str(m)
		model_path = Path(logdir, model_name, 'model')
		model_path_3.append(model_path)

	model0 = tf.keras.models.load_model(model_path_3[0])
	model1 = tf.keras.models.load_model(model_path_3[1])
	model2 = tf.keras.models.load_model(model_path_3[2])
	y_pred_NN_all0 = np.argmax(model0.predict(data_all), axis=1)
	y_pred_NN_all1 = np.argmax(model1.predict(data_all), axis=1)
	y_pred_NN_all2 = np.argmax(model2.predict(data_all), axis=1)

	for j in range(num_repeat):
		# cusum
		ind = range(N_test * j, N_test * (j + 1))
		y_test = y_all[ind, 0]
		y_pred_cusum = y_pred_cusum_all[ind]
		confusion_mtx = tf.math.confusion_matrix(y_test, y_pred_cusum)
		mer_cusum = (confusion_mtx[0, 1] + confusion_mtx[1, 0]) / N_test
		result_cusum[i, j, 0] = mer_cusum
		result_cusum[i, j, 1:] = np.reshape(confusion_mtx, (4,))
		# print("MER of CUSUM:", np.array(mer_cusum))
		# NN0
		y_pred_nn0 = y_pred_NN_all0[ind]
		confusion_mtx = tf.math.confusion_matrix(y_test, y_pred_nn0)
		mer_nn0 = (confusion_mtx[0, 1] + confusion_mtx[1, 0]) / N_test
		result_nn0[i, j, 0] = mer_nn0
		result_nn0[i, j, 1:] = np.reshape(confusion_mtx, (4,))
		# print("MER of NN0:", np.array(mer_nn0))
		# NN1
		y_pred_nn1 = y_pred_NN_all1[ind]
		confusion_mtx = tf.math.confusion_matrix(y_test, y_pred_nn1)
		mer_nn1 = (confusion_mtx[0, 1] + confusion_mtx[1, 0]) / N_test
		result_nn1[i, j, 0] = mer_nn1
		result_nn1[i, j, 1:] = np.reshape(confusion_mtx, (4,))
		# print("MER of NN1:", np.array(mer_nn1))
		# NN2
		y_pred_nn2 = y_pred_NN_all2[ind]
		confusion_mtx = tf.math.confusion_matrix(y_test, y_pred_nn2)
		mer_nn2 = (confusion_mtx[0, 1] + confusion_mtx[1, 0]) / N_test
		result_nn2[i, j, 0] = mer_nn2
		result_nn2[i, j, 1:] = np.reshape(confusion_mtx, (4,))
		# print("MER of NN2:", np.array(mer_nn2))

# %%
# save the cusum threshold to folder datasets/CusumResult/
cusum_vec = np.mean(result_cusum, axis=1, keepdims=False)[:, 0]
nn0_vec = np.mean(result_nn0, axis=1, keepdims=False)[:, 0]
nn1_vec = np.mean(result_nn1, axis=1, keepdims=False)[:, 0]
nn2_vec = np.mean(result_nn2, axis=1, keepdims=False)[:, 0]
mean_mer = np.array([cusum_vec, nn0_vec, nn1_vec, nn2_vec])
plt.figure(figsize=(10, 8))
markers = ['o', 'v', '*', 'd']
for i in range(4):
	plt.plot(
		N_vec, mean_mer[i, :], linewidth=4, marker=markers[i], markersize=14
	)

plt.legend(['CUSUM', r"$m_1$", r"$m_2$", r"$m_3$"], fontsize=18)
plt.xlabel('N', fontsize=25)
plt.ylabel('MER Average', fontsize=25)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)

cm_name = current_file + 'MERAverage.png'
latex_figures_folder = Path(file_path.parents[1], "Figures")
figure_path = Path(latex_figures_folder, cm_name)
plt.savefig(figure_path, format='png')

print('CUSUM', cusum_vec)
print('NN0', nn0_vec)
print('NN1', nn1_vec)
print('NN2', nn2_vec)
path_cusum = Path(cusum_result_folder, current_file + "result_cusum")
path_nn0 = Path(cusum_result_folder, current_file + "result_nn0")
path_nn1 = Path(cusum_result_folder, current_file + "result_nn1")
path_nn2 = Path(cusum_result_folder, current_file + "result_nn2")
np.save(path_cusum, result_cusum)
np.save(path_nn0, result_nn0)
np.save(path_nn1, result_nn1)
np.save(path_nn2, result_nn2)

# %%
