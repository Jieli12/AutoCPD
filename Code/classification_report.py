"""
Author         : Jie Li, Department of Statistics, London School of Economics.
Date           : 2022-11-01 10:53:56
Last Revision  : 2022-11-01 19:47:31
Last Author    : Jie Li
File Path      : /AutoCPD/Code/classification_report.py
Description    :








Copyright (c) 2022 by Jie Li, j.li196@lse.ac.uk
All Rights Reserved.
"""
import array_to_latex as a2l
import numpy as np
from sklearn.metrics import classification_report

n = 1000
r = 30
# load the classification data
datapath = "../datasets/BIC/"
fname_ypred_weak_LR = datapath + "y_pred_allbic_rweak_rep30.npy"
ypred_weak_LR = np.load(fname_ypred_weak_LR)
fname_ypred_strong_LR = datapath + "y_pred_allbic_rstrong_rep30.npy"
ypred_strong_LR = np.load(fname_ypred_strong_LR)

fname_ypred_weak_NN = datapath + "y_weak_pred_all_rep30.npy"
ypred_weak_NN = np.load(fname_ypred_weak_NN)
fname_ypred_strong_NN = datapath + "y_strong_pred_all_rep30.npy"
ypred_strong_NN = np.load(fname_ypred_strong_NN)

fname_y_test_weak = datapath + "y_test_rweak_rep30.npy"
y_test_weak = np.load(fname_y_test_weak)
fname_y_test_strong = datapath + "y_test_rstrong_rep30.npy"
y_test_strong = np.load(fname_y_test_strong)

recall_rep30 = np.zeros((6, 4, 30))
target_names = ['class 1', 'class 2', 'class 3', 'class 4', 'class 5']
for i in range(r):
	ind_temp = range(i * n, (i + 1) * n)
	y_test_weak_temp = y_test_weak[ind_temp,]
	y_test_strong_temp = y_test_strong[ind_temp,]
	ypred_weak_LR_temp = ypred_weak_LR[:, i]
	ypred_strong_LR_temp = ypred_strong_LR[:, i]
	ypred_weak_NN_temp = ypred_weak_NN[:, i]
	ypred_strong_NN_temp = ypred_strong_NN[:, i]
	# Weak LR
	result_temp = classification_report(
		y_test_weak_temp,
		ypred_weak_LR_temp,
		target_names=target_names,
		digits=4,
		output_dict=True
	)
	recall_rep30[0:5, 0, i] = [result_temp[j]['recall'] for j in target_names]
	recall_rep30[5, 0, i] = result_temp['accuracy']
	# Weak NN
	result_temp = classification_report(
		y_test_weak_temp,
		ypred_weak_NN_temp,
		target_names=target_names,
		digits=4,
		output_dict=True
	)
	recall_rep30[0:5, 1, i] = [result_temp[j]['recall'] for j in target_names]
	recall_rep30[5, 1, i] = result_temp['accuracy']
	# Strong LR
	result_temp = classification_report(
		y_test_strong_temp,
		ypred_strong_LR_temp,
		target_names=target_names,
		digits=4,
		output_dict=True
	)
	recall_rep30[0:5, 2, i] = [result_temp[j]['recall'] for j in target_names]
	recall_rep30[5, 2, i] = result_temp['accuracy']
	# Strong NN
	result_temp = classification_report(
		y_test_strong_temp,
		ypred_strong_NN_temp,
		target_names=target_names,
		digits=4,
		output_dict=True
	)
	recall_rep30[0:5, 3, i] = [result_temp[j]['recall'] for j in target_names]
	recall_rep30[5, 3, i] = result_temp['accuracy']

table = np.mean(recall_rep30, axis=2, keepdims=False)
a2l.to_clp(table, frmt='{:6.4f}', arraytype='bmatrix')
