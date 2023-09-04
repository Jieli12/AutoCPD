"""
Author         : Jie Li, Department of Statistics, London School of Economics.
Date           : 2023-05-08 20:31:26
Last Revision  : 2023-09-04 16:26:13
Last Author    : Jie Li
File Path      : /AutoCPD/Code/DS1ReportWilcox.py
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
rmse_weak = np.load("../datasets/CusumResult/DS1Weak.npy")
print(np.round(rmse_weak, 2))
rmse_weak_mosum = np.load("../datasets/CusumResult/DS1WeakMosumAdjust.npy")
rmse_weak[:, 1] = rmse_weak_mosum[:, 0]
print(np.round(rmse_weak, 2))
rmse_weak = rmse_weak[:, [0, 1, 3]]

plt.figure(figsize=(10, 8))
for j in range(3):
	plt.plot(
		n_candidates,
		rmse_weak[:, j],
		linewidth=4,
		marker=markers[j],
		markersize=14
	)
# plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
plt.legend(
	['CUSUM', 'MOSUM', 'Alg. 1'],
	fontsize=20,  # bbox_to_anchor=(1.04, 0.5),
	# loc="center left"
)
plt.xlabel('n', fontsize=25)
plt.ylabel('RMSE', fontsize=25)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
cm_name = 'S1Weak' + 'RMSEAverageW.eps'
# %%
# for med
rmse_strong = np.load("../datasets/CusumResult/DS1Strong.npy")
print(np.round(rmse_strong, 2))
rmse_strong_mosum = np.load("../datasets/CusumResult/DS1StrongMosumAdjust.npy")
rmse_strong[:, 1] = rmse_strong_mosum[:, 0]
print(np.round(rmse_strong, 2))
rmse_strong = rmse_strong[:, [0, 1, 3]]
plt.figure(figsize=(10, 8))
for j in range(3):
	plt.plot(
		n_candidates,
		rmse_strong[:, j],
		linewidth=4,
		marker=markers[j],
		markersize=14
	)
# plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
plt.legend(
	['CUSUM', 'MOSUM', 'Alg. 1'],
	fontsize=20,  # bbox_to_anchor=(1.04, 0.5),
	# loc="center left"
)
plt.xlabel('n', fontsize=25)
plt.ylabel('RMSE', fontsize=25)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
cm_name = 'S1Strong' + 'RMSEAverageW.eps'
# %%
