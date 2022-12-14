"""
Author         : Jie Li, Department of Statistics, London School of Economics.
Date           : 2022-10-18 16:01:59
Last Revision  : 2022-10-30 08:08:55
Last Author    : Jie Li
File Path      : /AutoCPD/Code/DataSetGen.py
Description    :








Copyright (c) 2022 by Jie Li, j.li196@lse.ac.uk
All Rights Reserved.
"""
import numpy as np
from sklearn.model_selection import train_test_split

from utils import *


# %%
def DataSetGen0(N_sub, n, mean_arg, var_arg, slope_arg, n_trim, seed=2022):
	np.random.seed(seed)  # numpy seed fixing
	sigma_1 = mean_arg[0]  # the standard deviation of noise
	upper_bound_mean = mean_arg[1]
	lower_bound_mean = mean_arg[2]
	upper_bound_abs_mean_difference = mean_arg[3]
	lower_bound_abs_mean_difference = mean_arg[4]
	mu_1all = np.zeros((N_sub, 2))
	i = 0
	while i < N_sub:
		temp = np.random.uniform(lower_bound_mean, upper_bound_mean, 2)
		abs_diff = np.abs(temp[1] - temp[0])
		if abs_diff >= lower_bound_abs_mean_difference and abs_diff <= upper_bound_abs_mean_difference:
			mu_1all[i, :] = temp
			i = i + 1

	mu_0all = np.column_stack((mu_1all[:, 0], np.repeat(0, N_sub)))
	mu_para = np.concatenate((mu_0all, mu_1all))

	n_range = np.arange(n_trim + 1, n - n_trim + 1)
	cp_mean = np.random.choice(n_range, (N_sub,))
	x_mean_0 = np.zeros((N_sub, n))
	x_mean_1 = np.zeros((N_sub, n))
	for i in range(N_sub):
		mu_1 = mu_1all[i, :]
		x_mean_0[i, :] = GenDataMean(1, n, cp=None, mu=mu_1, sigma=sigma_1)
		x_mean_1[
			i, :] = GenDataMean(1, n, cp=cp_mean[i], mu=mu_1, sigma=sigma_1)

	# one can manually change the parameters below to control the signal to noise ratio
	mu_2 = var_arg[0]
	upper_bound_std = var_arg[1]
	lower_bound_std = var_arg[2]
	upper_bound_abs_std_difference = var_arg[3]
	lower_bound_abs_std_difference = var_arg[4]
	sigma_2all = np.zeros((N_sub, 2))
	i = 0
	while i < N_sub:
		temp = np.random.uniform(lower_bound_std, upper_bound_std, 2)
		abs_diff = np.abs(temp[1] - temp[0])
		if abs_diff >= lower_bound_abs_std_difference and abs_diff <= upper_bound_abs_std_difference:
			sigma_2all[i, :] = temp
			i = i + 1

	sigma_0all = np.column_stack((sigma_2all[:, 0], np.repeat(0, N_sub)))
	sigma_para = np.concatenate((sigma_0all, sigma_2all))

	cp_var = np.random.choice(n_range, (N_sub,))
	x_var_0 = np.zeros((N_sub, n))
	x_var_1 = np.zeros((N_sub, n))
	for i in range(N_sub):
		sigma_2 = sigma_2all[i, :]
		x_var_0[i, :] = GenDataVariance(1, n, cp=None, mu=mu_2, sigma=sigma_2)
		x_var_1[
			i, :] = GenDataVariance(1, n, cp=cp_var[i], mu=mu_2, sigma=sigma_2)

	# one can manually change the parameters below to control the signal to noise ratio
	sigma_2 = slope_arg[0]
	upper_bound_slope = slope_arg[1]
	lower_bound_slope = slope_arg[2]
	upper_bound_abs_slope_difference = slope_arg[3]
	lower_bound_abs_slope_difference = slope_arg[4]
	slopes_all = np.zeros((N_sub, 2))
	i = 0
	while i < N_sub:
		temp = np.random.uniform(lower_bound_slope, upper_bound_slope, 2)
		abs_diff = np.abs(temp[1] - temp[0])
		if abs_diff >= lower_bound_abs_slope_difference and abs_diff <= upper_bound_abs_slope_difference:
			slopes_all[i, :] = temp
			i = i + 1

	slopes_0all = np.column_stack((slopes_all[:, 0], np.repeat(0, N_sub)))
	slopes_para = np.concatenate((slopes_0all, slopes_all))

	cp_slope = np.random.choice(n_range, (N_sub,))
	x_slope_0 = np.zeros((N_sub, n))
	x_slope_1 = np.zeros((N_sub, n))
	for i in range(N_sub):
		slopes = slopes_all[i, :]
		x_slope_0[i, :] = GenDataSlope(
			1, n, cp=None, slopes=slopes, sigma=sigma_2, start=0
		)
		x_slope_1[i, :] = GenDataSlope(
			1, n, cp=cp_slope[i], slopes=slopes, sigma=sigma_2, start=0
		)

	# concatenating
	data_x = np.concatenate(
		(x_mean_0, x_mean_1, x_var_1, x_slope_0, x_slope_1), axis=0
	)
	return {
		"data_x": data_x,
		"cp_mean": cp_mean,
		"cp_var": cp_var,
		"cp_slope": cp_slope,
		"mu_para": mu_para,
		"sigma_para": sigma_para,
		"slopes_para": slopes_para
	}
