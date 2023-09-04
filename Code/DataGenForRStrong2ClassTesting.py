"""
Author         : Jie Li, Department of Statistics, London School of Economics.
Date           : 2023-05-02 23:53:30
Last Revision  : 2023-09-04 11:22:07
Last Author    : Jie Li
File Path      : /AutoCPD/Code/DataGenForRStrong2ClassTesting.py
Description    :


2 class, test for strong SNR

Table R1, strong, ResNet, double-checked



Copyright (c) 2023 by Jie Li, j.li196@lse.ac.uk
All Rights Reserved.
"""
import pathlib
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from DataSetGen import *
from sklearn.metrics import auc, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from utils import *

np.random.seed(2022)  # numpy seed fixing
rep = 30
N_sub = 250 * rep
n = 400
n_trim = 40
mean_arg = np.array([0.7, 5, -5, 1.2, 0.6])
var_arg = np.array([0, 0.7, 0.3, 0.4, 0.2])
dataset = DataSetGenSim(N_sub, n, mean_arg, var_arg, n_trim)
data_x = dataset["data_x"]
data_y = np.repeat((0, 1), N_sub)
x_train, y_train = shuffle(data_x, data_y, random_state=42)
# %%
# LR method
opt_th = np.load("../datasets/CusumResult/opt_th_mean_var_strong.npy")

# Both
opt_th_meanvar = opt_th[0]
y_cusum_train_max = np.apply_along_axis(MaxCUSUM, 1, x_train)
y_meanvar_train_max = np.apply_along_axis(ComputeMeanVarNorm, 1, x_train)
y_pred_meanvar_train = y_meanvar_train_max > opt_th_meanvar
conf_mat = confusion_matrix(y_train, y_pred_meanvar_train)

path_th = "../datasets/CusumResult/acc_mean_var_strong30"
np.save(
	path_th,
	np.array(
		[
			# [n00 / N_sub, n11 / N_sub, 0.5 * (n00 / N_sub + n11 / N_sub)],
			[
				conf_mat[0, 0] / N_sub,
				conf_mat[1, 1] / N_sub,
				(conf_mat[0, 0] + conf_mat[1, 1]) / N_sub / 2
			]
		]
	)
)
# %%
# NN
data_x = Transform2D2TR(data_x, rescale=True, times=3)
x_test, y_test = shuffle(data_x, data_y, random_state=42)

current_file = Path(__file__).stem
print(current_file)
model_name = "StrongRevision2Class"
logdir = Path("tensorboard_logs", "Trial")
model_path = Path(logdir, model_name, 'model')
model = tf.keras.models.load_model(model_path)
model.summary()

# model_pred = model.evaluate(x_test, y_test, verbose=2)
y_pred = np.argmax(model.predict(x_test), axis=1)
confusion_mtx = tf.math.confusion_matrix(y_test, y_pred)
acc = np.sum(np.diag(confusion_mtx)) / N_sub / 2
acc0 = confusion_mtx[0, 0] / N_sub
acc1 = confusion_mtx[1, 1] / N_sub

path_th = "../datasets/CusumResult/acc_nn21_strong30"
np.save(path_th, np.array([acc0, acc1, acc]))
