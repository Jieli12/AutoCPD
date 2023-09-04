"""
Author         : Jie Li, Department of Statistics, London School of Economics.
Date           : 2023-05-08 21:02:11
Last Revision  : 2023-09-04 16:36:50
Last Author    : Jie Li
File Path      : /AutoCPD/Code/DS3ReportWilcox.py
Description    :








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
from DataSetGen import *
from keras import layers, losses, metrics, models
from matplotlib.ticker import StrMethodFormatter
from sklearn.metrics import auc, mean_squared_error, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from utils import *

# %% parameter settings
file_path = Path(__file__)
n_candidates = np.array(
	[300, 400, 500, 600], dtype=np.int32
)  # the length of time series
markers = ['o', 'v', 'X', '*', 's', 'D']
latex_figures_folder = Path(
	file_path.parents[1], "Latex", "JRSSB-Discussion-Manuscript", "figures"
)
# %%
# for weak
rmse_weak = np.load("../datasets/CusumResult/DS3Weak.npy")
print(np.round(rmse_weak, 2))
rmse_weak_mosum = np.load("../datasets/CusumResult/DS3WeakMosumAdjust.npy")
rmse_weak_wilcox = np.load("../datasets/RWilcoxon/DS3WeakRMSE.npy")
rmse_weak = np.column_stack([rmse_weak, rmse_weak_wilcox])
rmse_weak[:, 1] = rmse_weak_mosum[:, 0]
print(np.round(rmse_weak, 2))
rmse_weak = np.delete(rmse_weak, [2, 4], axis=1)

plt.figure(figsize=(10, 8))
for j in range(4):
	plt.plot(
		n_candidates,
		rmse_weak[:, j],
		linewidth=4,
		marker=markers[j],
		markersize=14
	)
# plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
plt.legend(
	['CUSUM', 'MOSUM', 'Alg. 1', 'Wilcoxon'],
	fontsize=20,  # bbox_to_anchor=(1.04, 0.5),
	loc=(0.25, 0.58)
)
plt.xlabel('n', fontsize=25)
plt.ylabel('RMSE', fontsize=25)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
cm_name = 'S3Weak' + 'RMSEAverageW.eps'
# %%
# for med
rmse_strong = np.load("../datasets/CusumResult/DS3Strong.npy")
print(np.round(rmse_strong, 2))
rmse_strong_mosum = np.load("../datasets/CusumResult/DS3StrongMosumAdjust.npy")
rmse_strong[:, 1] = rmse_strong_mosum[:, 0]
rmse_strong_wilcox = np.load("../datasets/RWilcoxon/DS3StrongRMSE.npy")
rmse_strong = np.column_stack([rmse_strong, rmse_strong_wilcox])
print(np.round(rmse_strong, 2))
rmse_strong = np.delete(rmse_strong, [2, 4], axis=1)

plt.figure(figsize=(10, 8))
for j in range(4):
	plt.plot(
		n_candidates,
		rmse_strong[:, j],
		linewidth=4,
		marker=markers[j],
		markersize=14
	)
# plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
plt.legend(
	['CUSUM', 'MOSUM', 'Alg. 1', 'Wilcoxon'],
	fontsize=20,  # bbox_to_anchor=(1.04, 0.5),
	loc=(0.25, 0.5)
)
plt.xlabel('n', fontsize=25)
plt.ylabel('RMSE', fontsize=25)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
cm_name = 'S3Strong' + 'RMSEAverageW.eps'

# %%
