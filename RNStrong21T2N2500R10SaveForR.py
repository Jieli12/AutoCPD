"""
Author         : Jie Li, Department of Statistics, London School of Economics.
Date           : 2023-05-18 10:34:22
Last Revision  : 2023-05-28 23:10:34
Last Author    : Jie Li
File Path      : /AI-assisstedChangePointDetection/Python/RNStrong21T2N2500R10SaveForR.py
Description    :

Data generation for LR-test in R

Table 1 of main text, double-checked.






Copyright (c) 2023 by Jie Li, j.li196@lse.ac.uk
All Rights Reserved.
"""

# %%

import pathlib
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from DataSetGen import *
from keras import layers, losses, metrics, models
# %%
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import train_test_split
from utilsMultimode import *

# %%
# set the random seed
np.random.seed(2022)  # numpy seed fixing
tf.random.set_seed(2022)  # tensorflow seed fixing
N_sub = 2500
n = 400
n_trim = 40
mean_arg = np.array([0.7, 5, -5, 1.2, 0.6])
var_arg = np.array([0, 0.7, 0.3, 0.4, 0.2])
slope_arg = np.array([0.5, 0.025, -0.025, 0.03, 0.015])

dataset = DataSetGen0(N_sub, n, mean_arg, var_arg, slope_arg, n_trim)

data_x = dataset["data_x"]
mu_para = dataset["mu_para"]
sigma_para = dataset["sigma_para"]
slopes_para = dataset["slopes_para"]
cp_var = dataset["cp_var"]
cp_slope = dataset["cp_slope"]
cp_mean = dataset["cp_mean"]
# %% normalization
# data_x = Transform2D2TR(data_x, rescale=True, times=5)
num_dataset = 5
labels = [0, 1, 2, 3, 4]
num_classes = len(set(labels))
data_y = np.repeat(labels, N_sub).reshape((N_sub * num_dataset, 1))
cp_non = np.zeros((N_sub,))
cp_all = np.concatenate((cp_non, cp_mean, cp_var, cp_non, cp_slope))
range = np.arange(N_sub * num_dataset)
x_train, x_test, y_train, y_test, cp_train, cp_test, ind_train, ind_test= train_test_split(
	data_x, data_y, cp_all, range, train_size=0.8, random_state=42
)
# %%
datapath = "../datasets/BICRevision/"
data_xpath = datapath + "RNStrong21R10data_x"
np.save(data_xpath, data_x)
cpt_varpath = datapath + "RNStrong21R10cpt_var"
np.save(cpt_varpath, cp_var)
cpt_slopepath = datapath + "RNStrong21R10cpt_slope"
np.save(cpt_slopepath, cp_slope)
cpt_meanpath = datapath + "RNStrong21R10cpt_mean"
np.save(cpt_meanpath, cp_mean)
data_x_testpath = datapath + "RNStrong21R10data_x_test"
np.save(data_x_testpath, x_test)
data_y_testpath = datapath + "RNStrong21R10data_y_test"
np.save(data_y_testpath, y_test)
