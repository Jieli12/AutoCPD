"""
Author         : Jie Li, Department of Statistics, London School of Economics.
Date           : 2022-10-29 22:38:25
Last Revision  : 2022-10-29 23:11:40
Last Author    : Jie Li
File Path      : /AutoCPD/Code/BICStrongPlot.py
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
from sklearn.model_selection import train_test_split

# %%

# set the random seed
np.random.seed(2022)  # numpy seed fixing
tf.random.set_seed(2022)  # tensorflow seed fixing
N_sub = 1000
n = 400
num_classes = 5
datapath = "../datasets/BIC/"

# %%
# only use BIC
fname_y_pred_bic = datapath + "y_pred_bic_rstrong" + ".npy"
fname_y_test = datapath + "y_test_rstrong" + ".npy"
y_pred_bic = np.load(fname_y_pred_bic)
y_test = np.load(fname_y_test)

confusion_mtx = tf.math.confusion_matrix(y_test, y_pred_bic)
plt.figure(figsize=(10, 8))
label_vec = range(0, num_classes)
sns.heatmap(
	confusion_mtx,
	cmap="YlGnBu",
	xticklabels=label_vec,
	yticklabels=label_vec,
	annot=True,
	fmt='g'
)
plt.xlabel('Prediction', fontsize=20)
plt.ylabel('Label', fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
acc = sum(np.diag(confusion_mtx)) / np.sum(confusion_mtx)
print(acc)
cm_nameeps = datapath + 'Confusion_matrixBIC_strong' + '.eps'
cm_namepng = datapath + 'Confusion_matrixBIC_strong' + '.png'
plt.savefig(cm_nameeps, format='eps')
plt.savefig(cm_namepng, format='png')

# %%
