"""
Author         : Jie Li, Department of Statistics, London School of Economics.
Date           : 2023-04-12 08:37:27
Last Revision  : 2023-09-04 13:12:29
Last Author    : Jie Li
File Path      : /AutoCPD/Code/S3R0PredictPlot.py
Description    :

Scenario 3, rho=0, hidden layer=1, number of threshold values=6

Figure R4, double checked




Copyright (c) 2023 by Jie Li, j.li196@lse.ac.uk
All Rights Reserved.
"""
# %%

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import StrMethodFormatter

file_path = Path(__file__)
result_cusum = np.load('../datasets/CusumResult/S3R0Trainresult_cusumT.npy')
result_wilcox = np.load('../datasets/CusumResult/S3R0Trainresult_wilcoxT.npy')
result_nn = np.load('../datasets/CusumResult/S3R0Trainresult_nnT.npy')
N_vec = np.arange(100, 1100, 100)
cusum_mat = np.mean(result_cusum, axis=1, keepdims=False)[:, 0]
wilcox_mat = np.mean(result_wilcox, axis=1, keepdims=False)[:, 0]
nn_mat = np.mean(result_nn, axis=2, keepdims=False)[:, :, 0]
markers = ['o', 'v', 'd', 's', 'X', '*', 'p']
lr = np.column_stack([cusum_mat, wilcox_mat])
mer = np.concatenate([lr, nn_mat], axis=1)
latex_figures_folder = Path(
	file_path.parents[1], "Latex", "JRSSB-Discussion-Manuscript", "figures"
)
# %%
plt.figure(figsize=(10, 8))
for i in [0, 1, 2, 6]:
	mer_i = mer[:, i]
	plt.plot(N_vec, mer_i, linewidth=4, marker=markers[i], markersize=14)

plt.legend(
	['CUSUM', "Wilcoxon", r"$m^{(2)}$,L=1", r"$m^{(2)}$,L=1, Z=3"],
	fontsize=16,
	# bbox_to_anchor=(1.04, 0.5),
	loc=(0.06, 0.4)
)

plt.xlabel('N', fontsize=25)
plt.ylabel('MER Average', fontsize=25)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
