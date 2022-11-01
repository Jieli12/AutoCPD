"""
Author         : Jie Li, Department of Statistics, London School of Economics.
Date           : 2022-10-30 16:44:10
Last Revision  : 2022-11-01 19:40:52
Last Author    : Jie Li
File Path      : /AutoCPD/Code/ResNetWeakRep30.py
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
import seaborn as sns
import tensorflow as tf
import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import tensorflow_docs.plots
from keras import layers, losses, metrics, models
# %%
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import train_test_split

from DataSetGen import *
from utils import *

# %%
# set the random seed
np.random.seed(2022)  # numpy seed fixing
tf.random.set_seed(2022)  # tensorflow seed fixing
datapath = "../datasets/BIC/"
fname_x_test_r = datapath + "x_test_allweak_rep30.npy"
x_test = np.load(fname_x_test_r)
fname_y_test = datapath + "y_test_rweak_rep30.npy"
y_test = np.load(fname_y_test)

current_file = Path(__file__).stem
print(current_file)
model_name = current_file
logdir = Path("tensorboard_logs", "Trial")
model_name = 'ResNetN1kE8tanhDecay10kScale'
model_path = Path(logdir, model_name, 'model')
model = tf.keras.models.load_model(model_path)
model.summary()

rep = 30
n = 1000
acc = np.zeros((rep,))
pred_all = np.zeros((n, rep))
for i in range(rep):
	ind_temp = range(i * n, (i + 1) * n)
	x_test_temp = x_test[ind_temp, :, :]
	y_test_temp = y_test[ind_temp,]
	model_pred = model.evaluate(x_test_temp, y_test_temp, verbose=2)
	y_pred_temp = np.argmax(model.predict(x_test_temp), axis=1)
	confusion_mtx = tf.math.confusion_matrix(y_test_temp, y_pred_temp)
	acc[i] = np.sum(np.diag(confusion_mtx)) / n
	pred_all[:, i] = y_pred_temp

np.mean(acc, axis=0)
fname_y_weak_acc = datapath + "y_weak_rep30.npy"
np.save(fname_y_weak_acc, acc)
fname_y_weak_pred_all = datapath + "y_weak_pred_all_rep30.npy"
np.save(fname_y_weak_pred_all, pred_all)
