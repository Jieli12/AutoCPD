"""
Author         : Jie Li, Department of Statistics, London School of Economics.
Date           : 2022-10-23 09:53:44
Last Revision  : 2022-10-29 23:08:51
Last Author    : Jie Li
File Path      : /AutoCPD/Code/ScenarioS2RhoRandomTrain.py
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

from utils import *

# %% parameter settings
n_vec = np.array([100], dtype=np.int32)  # the length of time series
n_len = len(n_vec)
epsilon = 0.05
Q = np.array(np.floor(np.log2(n_vec / 2)), dtype=np.int32) + 1
# the number of hidden nodes
m_mat = np.c_[3 * np.ones((n_len,), dtype=np.int32), 4 * Q, 2 * n_vec - 2]
N_vec = np.arange(100, 1100, 100, dtype=np.int32)  # the sample size
num_N = len(N_vec)
B = np.sqrt(8 * np.log(n_vec / epsilon) / n_vec)
mu_L = 0
tau_bound = 2
sigma = np.sqrt(2)
B_bound = np.array([0.5, 1.5])
# parameters for neural network
learning_rate = 1e-3
epochs = 200
batch_size = 32
num_classes = 2

#  setup the tensorboard
file_path = Path(__file__)
cusum_result_folder = Path(file_path.parents[1], "datasets", "CusumResult")
current_file = file_path.stem
print(current_file)
logdir = Path("tensorboard_logs", "Trial")

num_models = n_len * num_N
d = {
	'n': np.repeat(0, num_models),
	'N': np.repeat(0, num_models),
	'B': np.repeat(0.0, num_models),
	'Threshold': np.repeat(0.0, num_models)
}
Cusum_th = pd.DataFrame(data=d)

# %% main double for loop
num_loops = 0
for i in range(n_len):
	n = n_vec[i]
	print(n, i)
	for j in range(num_N):
		N = int(N_vec[j] / 2)
		#  generate the dataset for alternative hypothesis
		np.random.seed(2022)  # numpy seed fixing
		tf.random.set_seed(2022)  # tensorflow seed fixing
		result = DataGenAlternative(
			N_sub=N,
			B=B[i],
			mu_L=mu_L,
			n=n,
			tau_bound=tau_bound,
			B_bound=B_bound,
			type='ARrho',
			sigma=sigma
		)
		data_alt = result["data"]
		tau_alt = result["tau_alt"]
		mu_R_alt = result["mu_R_alt"]
		#  generate dataset for null hypothesis
		data_null = GenDataMeanARrho(
			N, n, cp=None, mu=(mu_L, mu_L), sigma=sigma
		)
		data_all = np.concatenate((data_alt, data_null), axis=0)
		y_all = np.repeat((1, 0), N).reshape((2 * N, 1))
		tau_all = np.concatenate((tau_alt, np.repeat(0, N)), axis=0)
		mu_R_all = np.concatenate((mu_R_alt, np.repeat(mu_L, N)), axis=0)
		#  generate the training dataset and test dataset
		x_train, y_train, tau_train, mu_R_train = shuffle(data_all, y_all, tau_all, mu_R_all, random_state=42)
		# CUSUM, find the optimal threshold based on the training dataset

		# Compute the theoretical threshold under iid case as a start.
		threshold_star_theoretical = np.sqrt(2 * np.log(n / epsilon))
		y_cusum_train_max = np.apply_along_axis(MaxCUSUM, 1, x_train)
		threshold_candidate = threshold_star_theoretical * np.arange(
			0.1, 3, 0.05
		)
		mer = np.repeat(0.0, len(threshold_candidate))
		for ind, th in enumerate(threshold_candidate):
			# print(i, th)
			y_pred_cusum_train = y_cusum_train_max > th
			conf_mat = tf.math.confusion_matrix(y_pred_cusum_train, y_train)
			mer[ind] = (conf_mat[0, 1] + conf_mat[1, 0]) / N / 2

		threshold_opt = threshold_candidate[np.argmin(mer)]
		Cusum_th.at[num_loops, 'n'] = n
		Cusum_th.at[num_loops, 'N'] = 2 * N
		Cusum_th.at[num_loops, 'B'] = B[i]
		Cusum_th.at[num_loops, 'Threshold'] = threshold_opt
		num_loops = num_loops + 1
		for k in range(3):
			m = m_mat[i, k]
			model_name = current_file + "n" + str(n) + "N" + str(
				2 * N
			) + "m" + str(m)
			print(model_name)
			# build the model
			input = layers.Input(shape=(n,), name="Input")
			x = layers.Dense(m, activation="relu",
								kernel_regularizer='l2')(input)
			output = layers.Dense(num_classes)(x)
			model = models.Model(input, output, name=model_name)
			model.summary()
			# build the model, train and save it to disk
			lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
				learning_rate, decay_steps=10000, decay_rate=1, staircase=False
			)

			def get_optimizer():
				return tf.keras.optimizers.Adam(lr_schedule)

			def get_callbacks(name):
				name1 = name + '/log.csv'
				return [
					tfdocs.modeling.EpochDots(),
					tf.keras.callbacks.EarlyStopping(
						monitor='val_sparse_categorical_crossentropy',
						patience=800,
						min_delta=1e-3
					),
					tf.keras.callbacks.TensorBoard(Path(logdir, name)),
					tf.keras.callbacks.CSVLogger(Path(logdir, name1)),
				]

			def compile_and_fit(model, name, optimizer=None, max_epochs=10000):
				if optimizer is None:
					optimizer = get_optimizer()
				model.compile(
					optimizer=optimizer,
					loss=losses.SparseCategoricalCrossentropy(from_logits=True),
					metrics=[
						metrics.SparseCategoricalCrossentropy(
							from_logits=True,
							name='sparse_categorical_crossentropy'
						),
						"accuracy"
					]
				)
				history = model.fit(
					x_train,
					y_train,
					epochs=max_epochs,
					batch_size=batch_size,
					validation_split=0.2,
					callbacks=get_callbacks(name),
					verbose=2
				)
				return history

			size_histories = {}
			size_histories[model_name] = compile_and_fit(
				model,
				model_name,
				# optimizer=optimizers.Adam(learning_rate=learning_rate),
				max_epochs=epochs
			)
			plotter = tfdocs.plots.HistoryPlotter(
				metric='accuracy', smoothing_std=10
			)
			plt.figure(figsize=(10, 8))
			plotter.plot(size_histories)
			acc_name = model_name + '+acc.png'
			acc_path = Path(logdir, model_name, acc_name)
			plt.savefig(acc_path)
			plt.clf()
			model_path = Path(logdir, model_name, 'model')
			model.save(model_path)

		plt.figure(figsize=(10, 8))
		plt.plot(mer)
		mer_name = model_name + '+grid_search.png'
		mer_path = Path(logdir, model_name, mer_name)
		plt.savefig(mer_path)
		plt.clf()
# %%
# save the cusum threshold to folder datasets/CusumResult/
pkl_path = Path(cusum_result_folder, current_file + '.pkl')
Cusum_th.to_pickle(pkl_path)
