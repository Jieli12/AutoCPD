"""
Author         : Jie Li, Department of Statistics, London School of Economics.
Date           : 2023-03-27 20:40:24
Last Revision  : 2023-09-04 14:30:02
Last Author    : Jie Li
File Path      : /AutoCPD/Code/SARho0ToOthersPredictReplot.py
Description    : Inheriting from SARho0ToOthersPredict.py, adjust the MER of cusum and replot.


Figure R3, double checked





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
import tensorflow as tf
import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import tensorflow_docs.plots
from DataSetGen import *
from keras import layers, losses, metrics, models
from matplotlib.ticker import StrMethodFormatter
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from utils import *

# %% parameter settings
N_vec = np.arange(100, 800, 100, dtype=np.int32)  # the sample size
file_path = Path(__file__)
cusum_result_folder = Path(file_path.parents[1], "datasets", "CusumResult")
current_file = file_path.stem
current_file = current_file.replace(
	'SARho0ToOthersPredictReplot', 'ScenarioARho0Train'
)
print(current_file)
# %%
path_nn1 = Path(cusum_result_folder, current_file + "A2OtherResult_nn1.npy")
path_nn2 = Path(cusum_result_folder, current_file + "A2OtherResult_nn2.npy")
path_nnL5 = Path(cusum_result_folder, current_file + "A2OtherResult_nnL5.npy")
path_nnL10 = Path(cusum_result_folder, current_file + "A2OtherResult_nnL10.npy")
result_nn1 = np.load(path_nn1)
result_nn2 = np.load(path_nn2)
result_nnL5 = np.load(path_nnL5)
result_nnL10 = np.load(path_nnL10)

# plot
nn1_vec = np.mean(result_nn1, axis=2, keepdims=False)[:, :, 0]
nn2_vec = np.mean(result_nn2, axis=2, keepdims=False)[:, :, 0]
nnL5_vec = np.mean(result_nnL5, axis=2, keepdims=False)[:, :, 0]
nnL10_vec = np.mean(result_nnL10, axis=2, keepdims=False)[:, :, 0]
cusum_A = np.load("../datasets/CusumResult/ScenarioARho0Trainresult_cusum.npy")
cusum_A07 = np.load(
	"../datasets/CusumResult/ScenarioARho07Trainresult_cusum.npy"
)
cusum_C = np.load("../datasets/CusumResult/ScenarioCRho0Trainresult_cusum.npy")
cusum_D = np.load(
	"../datasets/CusumResult/ScenarioDRhoRandomTrainresult_cusum.npy"
)
cusum_A = np.mean(cusum_A, axis=1, keepdims=False)[:, 0]
cusum_A07 = np.mean(cusum_A07, axis=1, keepdims=False)[:, 0]
cusum_C = np.mean(cusum_C, axis=1, keepdims=False)[:, 0]
cusum_D = np.mean(cusum_D, axis=1, keepdims=False)[:, 0]
cusum_vec = np.column_stack((cusum_A, cusum_A07, cusum_C[:7], cusum_D[:7]))

for k in range(4):
	mean_mer = np.array(
		[
			cusum_vec[:, k],
			nn1_vec[:, k],
			nn2_vec[:, k],
			nnL5_vec[:, k],
			nnL10_vec[:, k]
		]
	)
	plt.figure(figsize=(10, 8))
	markers = ['o', 'v', 'd', 's', 'X']
	for i in range(5):
		plt.plot(
			N_vec,
			mean_mer[i, :],
			linewidth=4,
			marker=markers[i],
			markersize=14
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
		fontsize=20
	)
	plt.xlabel('N', fontsize=25)
	plt.ylabel('MER Average', fontsize=25)
	plt.xticks(fontsize=25)
	plt.yticks(fontsize=25)

	cm_name = current_file + 'MERAverageA2Othersk=' + str(k) + '.eps'
	latex_figures_folder = Path(
		file_path.parents[1], "Latex", "JRSSB-Discussion-Manuscript", "figures"
	)
	figure_path = Path(latex_figures_folder, cm_name)
	plt.savefig(figure_path, format='eps')
	plt.clf()
