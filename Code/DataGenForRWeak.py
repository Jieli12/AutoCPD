"""
Author         : Jie Li, Department of Statistics, London School of Economics.
Date           : 2022-10-29 22:07:30
Last Revision  : 2022-10-29 23:11:20
Last Author    : Jie Li
File Path      : /AutoCPD/Code/DataGenForRWeak.py
Description    :








Copyright (c) 2022 by Jie Li, j.li196@lse.ac.uk
All Rights Reserved.
"""
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from DataSetGen import *
from utils import *

np.random.seed(2022)  # numpy seed fixing
N_sub = 1000
n = 400
n_trim = 40
mean_arg = np.array([0.7, 5, -5, 0.5, 0.25])
var_arg = np.array([20, 0.7, 0.3, 0.25, 0.12])
slope_arg = np.array([0.5, 0.025, -0.025, 0.012, 0.005])

dataset = DataSetGen0(N_sub, n, mean_arg, var_arg, slope_arg, n_trim)

data_x = dataset["data_x"]
data_x = Transform2D(data_x, rescale=True, cumsum=False)
data_x = Standardize(data_x)

datamin = data_x.min(axis=2, keepdims=True)
datamax = data_x.max(axis=2, keepdims=True)
data_x = 2 * (data_x - datamin) / (datamax - datamin) - 1

num_dataset = 5
labels = [0, 1, 2, 3, 4]
num_classes = len(set(labels))
data_y = np.repeat(labels, N_sub).reshape((N_sub * num_dataset, 1))
cp_non = np.zeros((N_sub,))
range = np.arange(N_sub * num_dataset)
x_train, x_test, y_train, y_test, ind_train, ind_test= train_test_split(
	data_x, data_y, range, train_size=0.6, random_state=42
)

# save the dataset for R
datapath = "../datasets/BIC/"
fname_x_test_r = datapath + "x_test_rweak"
x_test_r = x_test[:, 0, :]
np.save(fname_x_test_r, x_test_r)
fname_y_test = datapath + "y_test_rweak"
np.save(fname_y_test, y_test)
