"""
Author         : Jie Li, Department of Statistics, London School of Economics.
Date           : 2022-11-05 09:01:18
Last Revision  : 2022-11-06 10:19:46
Last Author    : Jie Li
File Path      : /AutoCPD/Code/ScenarioS2RhoRandomMultiLPredictReplot.py
Description    :








Copyright (c) 2022 by Jie Li, j.li196@lse.ac.uk
All Rights Reserved.
"""

# %%

import pathlib
from pathlib import Path

import array_to_latex as a2l
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
from matplotlib.ticker import StrMethodFormatter
# %%
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from DataSetGen import *
from utils import *

# %% parameter settings
N_vec = np.arange(100, 1100, 100, dtype=np.int32)  # the sample size
#  setup the tensorboard
file_path = Path(__file__)
cusum_result_folder = Path(file_path.parents[1], "datasets", "CusumResult")
current_file = file_path.stem
current_file = current_file.replace('MultiLPredictReplot', '')
print(current_file)

# %%
path_cusum = Path(cusum_result_folder, current_file + "Trainresult_cusum.npy")
path_nnL10 = Path(
	cusum_result_folder, current_file + "L10Trainresult_nnL10.npy"
)
path_nnL5 = Path(cusum_result_folder, current_file + "L5Trainresult_nnL5.npy")
path_nn1 = Path(cusum_result_folder, current_file + "Trainresult_nn1.npy")
path_nn2 = Path(cusum_result_folder, current_file + "Trainresult_nn2.npy")
result_cusum = np.load(path_cusum)
result_nn1 = np.load(path_nn1)
result_nn2 = np.load(path_nn2)
result_nnL5 = np.load(path_nnL5)
result_nnL10 = np.load(path_nnL10)
cusum_vec = np.mean(result_cusum, axis=1, keepdims=False)[:, 0]
nn1_vec = np.mean(result_nn1, axis=1, keepdims=False)[:, 0]
nn2_vec = np.mean(result_nn2, axis=1, keepdims=False)[:, 0]
nnL5_vec = np.mean(result_nnL5, axis=1, keepdims=False)[:, 0]
nnL10_vec = np.mean(result_nnL10, axis=1, keepdims=False)[:, 0]
mean_mer = np.array([cusum_vec, nn1_vec, nn2_vec, nnL5_vec, nnL10_vec])
plt.figure(figsize=(10, 8))
markers = ['o', 'v', 'd', 's', 'X']
for i in range(5):
	plt.plot(
		N_vec, mean_mer[i, :], linewidth=4, marker=markers[i], markersize=14
	)

plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
plt.legend(
	[
		'CUSUM',
		r"$m^{(1)}$,L=1",
		r"$m^{(2)}$,L=1",
		r"$m^{(1)}$,L=5",
		r"$m^{(1)}$,L=10",
	],
	fontsize=25
)
# plt.ylim([0.24, 0.5])
plt.xlabel('N', fontsize=25)
plt.ylabel('MER Average', fontsize=25)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
cm_name = current_file + 'MERAverage5.png'
latex_figures_folder = Path(file_path.parents[1], "Figures")
figure_path = Path(latex_figures_folder, cm_name)
plt.savefig(figure_path, format='png')

# %%
