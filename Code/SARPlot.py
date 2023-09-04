"""
Author         : Jie Li, Department of Statistics, London School of Economics.
Date           : 2023-05-11 10:08:49
Last Revision  : 2023-09-04 15:05:57
Last Author    : Jie Li
File Path      : /AutoCPD/Code/SARPlot.py
Description    :


Figure R2 plot, double-checked





Copyright (c) 2023 by Jie Li, j.li196@lse.ac.uk
All Rights Reserved.
"""

# %%

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import StrMethodFormatter

file_path = Path(__file__)
result_nn = np.load('../datasets/CusumResult/SARTrainresult_nnAR.npy')
result_nn_square = np.load(
	'../datasets/CusumResult/SARSquareTrainresult_nnARsquare.npy'
)
result_nn_resnet = np.load(
	'../datasets/CusumResult/SARSquareResNetresult_nnAR.npy'
)
N_vec = np.arange(100, 800, 100)
nn_mat = np.mean(result_nn, axis=2, keepdims=False)[:, :, 0]
nn_mat_square = np.mean(result_nn_square, axis=2, keepdims=False)[:, :, 0]
nn_mat_resnet = np.mean(result_nn_resnet, axis=1, keepdims=False)[:, 0]
# %%
markers = ['o', 'v', 'X', '*']
nn_mat = np.column_stack([nn_mat, nn_mat_resnet])
latex_figures_folder = Path(
	file_path.parents[1], "Latex", "JRSSB-Discussion-Manuscript", "figures"
)

plt.figure(figsize=(10, 8))
plt.ylim(0.05, 0.4)
for j in range(0, 4):
	plt.plot(N_vec, nn_mat[:, j], linewidth=4, marker=markers[j], markersize=14)
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
plt.legend(
	[r"$m^{(1)}$,L=1", r"$m^{(1)}$,L=5", r"$m^{(2)}$,L=1", "ResNet"],
	fontsize=20
)
plt.xlabel('N', fontsize=25)
plt.ylabel('MER Average', fontsize=25)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)

# %%

plt.figure(figsize=(10, 8))
plt.ylim(0.05, 0.4)
for j in range(0, nn_mat_square.shape[1]):
	plt.plot(
		N_vec,
		nn_mat_square[:, j],
		linewidth=4,
		marker=markers[j],
		markersize=14
	)
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
plt.legend([r"$m^{(1)}$,L=1", r"$m^{(1)}$,L=5", r"$m^{(2)}$,L=1"], fontsize=20)
plt.xlabel('N', fontsize=25)
plt.ylabel('MER Average', fontsize=25)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)

# %%
