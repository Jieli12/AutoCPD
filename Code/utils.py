"""
Author         : Jie Li, Department of Statistics, London School of Economics.
Date           : 2022-01-12 15:19:50
Last Author    : Jie Li
Last Revision  : 2022-10-30 08:23:52
File Path      : /AutoCPD/Code/utils.py
Description    :  this script includes the utility function for multimode change points detection (single).








Copyright (c) 2022 by Jie Li, j.li196@lse.ac.uk
All Rights Reserved.
"""
import os
import pathlib
import posixpath
import warnings
from cmath import tanh
from re import I

import numpy as np
import pandas as pd
from keras import layers, losses, metrics, models
from scipy.stats import cauchy, linregress
from sklearn.isotonic import IsotonicRegression
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.nonparametric.kernel_regression import KernelReg
from statsmodels.tsa.arima_process import ArmaProcess


def GenDataMean(N, n, cp, mu, sigma):
	"""
	The function  generates the data for change in mean. When "cp" is None, it generates the data without change point.

	Parameters
	----------
	N : int
		the number of data
	n : int
		the number of variables
	cp : int
		the change point, only 1 change point is accepted in this function.
	mu : float
		the piecewise mean
	sigma : float
		the standard deviation

	Returns
	-------
	numpy array
		2D array with size (N, n)
	"""
	if cp is None:
		data = np.random.normal(mu[0], sigma, (N, n))
	else:
		data1 = np.random.normal(mu[0], sigma, (N, cp))
		data2 = np.random.normal(mu[1], sigma, (N, n - cp))
		data = np.concatenate((data1, data2), axis=1)
	return data


def GenDataMeanAR(N, n, cp, mu, sigma, coef):
	"""
	The function  generates the data for change in mean with AR(p) noise. When "cp" is None, it generates the data without change point.

	Parameters
	----------
	N : int
		the number of data
	n : int
		the number of variables
	cp : int
		the change point, only 1 change point is accepted in this function.
	mu : float
		the piecewise mean
	sigma : float
		the standard deviation
	coef : float array
		the coefficients of AR(p) model

	Returns
	-------
	numpy array
		2D array with size (N, n)
	"""
	arparams = coef
	maparams = np.array([1])
	ar_process = ArmaProcess(arparams, maparams)
	if cp is None:
		data = mu[0] + np.transpose(
			ar_process.generate_sample(nsample=(n, N), scale=sigma)
		)
	else:
		noise = ar_process.generate_sample(nsample=(n, N), scale=sigma)
		signal = np.repeat(mu, (cp, n - cp))
		data = np.transpose(noise) + signal
	return data


def GenDataMeanARH(N, n, cp, mu, coef, scale):
	"""
	The function  generates the data for change in mean with Cauchy noise with location parameter 0 and scale parameter 'scale'. When "cp" is None, it generates the data without change point.

	Parameters
	----------
	N : int
		the number of data
	n : int
		the number of variables
	cp : int
		the change point, only 1 change point is accepted in this function.
	mu : float
		the piecewise mean
	coef : float array
		the coefficients of AR(p) model
	scale : the scale parameter of Cauchy distribution
		the coefficients of AR(p) model

	Returns
	-------
	numpy array
		2D array with size (N, n)
	"""
	# initialize
	n1 = n + 30
	x_0 = np.ones((N,), dtype=np.float64)
	# eps_mat = np.random.standard_cauchy((N, n1))
	eps_mat = cauchy.rvs(loc=0, scale=scale, size=(N, n1))
	noise_mat = np.empty((N, n1))
	for i in range(n1):
		x_0 = coef * x_0 + eps_mat[:, i]
		noise_mat[:, i] = x_0

	if cp is None:
		data = mu[0] + noise_mat[:, -n:]
	else:
		signal = np.repeat(mu, (cp, n - cp))
		data = signal + noise_mat[:, -n:]

	return data


def GenDataMeanARrho(N, n, cp, mu, sigma):
	"""
	The function  generates the data for change in mean with AR(p) noise. The autoregressive coefficient is generated from standard uniform distribution. When "cp" is None, it generates the data without change point.

	Parameters
	----------
	N : int
		the number of data
	n : int
		the number of variables
	cp : int
		the change point, only 1 change point is accepted in this function.
	mu : float
		the piecewise mean
	sigma : float
		the standard variance of normal distribution

	Returns
	-------
	numpy array
		2D array with size (N, n)
	"""
	# initialize
	n1 = n + 30
	x_0 = np.ones((N,), dtype=np.float64)
	rho = np.random.uniform(0, 1, (N, n1))
	eps_mat = np.random.normal(0, sigma, size=(N, n1))
	noise_mat = np.empty((N, n1))
	for i in range(n1):
		x_0 = np.multiply(rho[:, i], x_0) + eps_mat[:, i]
		noise_mat[:, i] = x_0
	if cp is None:
		data = mu[0] + noise_mat[:, -n:]
	else:
		signal = np.repeat(mu, (cp, n - cp))
		data = signal + noise_mat[:, -n:]
	return data


def GenDataVariance(N, n, cp, mu, sigma):
	"""
	The function  generates the data for change in variance. When "cp" is None, it generates the data without change point in variance.

	Parameters
	----------
	N : int
		the number of data
	n : int
		the number of variables
	cp : int
		the change point, only 1 change point is accepted in this function.
	mu : float
		the piecewise mean
	sigma : float
		the standard deviation

	Returns
	-------
	numpy array
		2D array with size (N, n)
	"""
	if cp is None:
		data = np.random.normal(mu, sigma[0], (N, n))
	else:
		data1 = np.random.normal(mu, sigma[0], (N, cp))
		data2 = np.random.normal(mu, sigma[1], (N, n - cp))
		data = np.concatenate((data1, data2), axis=1)
	return data


def GenDataSlope(N, n, cp, slopes, sigma, start):
	"""
	The function  generates the data for change in slope. When "cp" is None, it generates the data without change point in slope.

	Parameters
	----------
	N : int
		the number of data
	n : int
		the number of variables
	cp : int
		the change point, only 1 change point is accepted in this function.
	slopes : float
		the slopes
	sigma : float
		the standard deviation
	start : float
		the y-coordinate of the start point

	Returns
	-------
	numpy array
		2D array with size (N, n)
	"""
	if cp is None:
		y = start + slopes[0] * np.arange(n)
		data = np.tile(y, (N, 1))  # repeat the row N times
	else:
		y1 = start + slopes[0] * np.arange(cp)
		y2 = y1[-1] + slopes[1] * np.arange(1, n - cp + 1)
		y = np.concatenate((y1, y2))
		data = np.tile(y, (N, 1))
	return data + np.random.normal(0, sigma, (N, n))


def Standardize(data):
	"""
	Data standardization

	Parameters
	----------
	data : numpy array
		the data set with size (N, ..., n)

	Returns
	-------
	data
		standardized data
	"""
	data = data.transpose()
	data = (data - data.mean(axis=0)) / data.std(axis=0)
	return data.transpose()


def Transform2D(data_y, rescale=False, cumsum=False):
	"""
	Apply 4 transformations (original, squared, log squared, tanh) to the same dataset

	Parameters
	----------
	data_y : numpy array
		the 2-D array
	rescale : logical bool
		default False
	cusum : logical bool
		replace tanh transformation with cusum transformation, default False
	Returns
	-------
	numpy array
		3-D arrary with size (N, 4,  n)
	"""
	N, n = data_y.shape
	data_y[data_y == 0.0] = 1e-8
	if rescale is True:
		data_y = data_y.transpose()
		m2 = np.square(data_y)
		m2log = np.log(m2)
		data_y = (data_y - data_y.min(axis=0)
					) / (data_y.max(axis=0) - data_y.min(axis=0))
		if cumsum is True:
			m_tanh = np.cumsum(data_y)
		else:
			m_tanh = np.tanh(data_y)

		m2 = (m2 - m2.min(axis=0)) / (m2.max(axis=0) - m2.min(axis=0))
		m2log = (m2log - m2log.min(axis=0
									)) / (m2log.max(axis=0) - m2log.min(axis=0))
		m_tanh = (m_tanh - m_tanh.min(axis=0)
					) / (m_tanh.max(axis=0) - m_tanh.min(axis=0))
		data_y = data_y.transpose()
		m2 = m2.transpose()
		m2log = m2log.transpose()
		m_tanh = m_tanh.transpose()
	else:
		m2 = np.square(data_y)
		m2log = np.log(m2)
		if cumsum is True:
			m_tanh = np.cumsum(data_y)
		else:
			m_tanh = np.tanh(data_y)

		# m_sin = np.sin(data_y)

	y_new = data_y.reshape((N, 1, n))
	m2 = m2.reshape((N, 1, n))
	m2log = m2log.reshape((N, 1, n))
	m_tanh = m_tanh.reshape((N, 1, n))
	# m_sin = m_sin.reshape((N, 1, n))
	return np.concatenate((y_new, m2, m2log, m_tanh), axis=1)


def plotsmooth(values, std):
	"""Smooths a list of values by convolving with a Gaussian distribution.
	Assumes equal spacing.
	Args:
		values: A 1D array of values to smooth.
		std: The standard deviation of the Gaussian distribution. The units are
		array elements.
	Returns:
		The smoothed array.
	"""
	width = std * 4
	x = np.linspace(-width, width, min(2 * width + 1, len(values)))
	kernel = np.exp(-(x / 5)**2)

	values = np.array(values)
	weights = np.ones_like(values)

	smoothed_values = np.convolve(values, kernel, mode='same')
	smoothed_weights = np.convolve(weights, kernel, mode='same')

	return smoothed_values / smoothed_weights


def resblock(x, kernel_size, filters, strides=1):
	x1 = layers.Conv2D(filters, kernel_size, strides=strides, padding='same')(x)
	x1 = layers.BatchNormalization()(x1)
	x1 = layers.ReLU()(x1)
	x1 = layers.Conv2D(filters, kernel_size, padding='same')(x1)
	x1 = layers.BatchNormalization()(x1)
	if strides != 1:
		x = layers.Conv2D(filters, 1, strides=strides, padding='same')(x)
		x = layers.BatchNormalization()(x)

	x1 = layers.Add()([x, x1])
	x1 = layers.ReLU()(x1)
	return x1


def labelTransition(data, label, ind, length, size, num_trim=100):
	s = label['start'][ind:ind + 2]
	e = label['end'][ind:ind + 2]
	state = label['state'][ind:ind + 2]
	new_label = state[ind] + "->" + state[ind + 1]
	logical0 = (data['time'] >= s[ind]) & (data['time'] <= e[ind])
	logical1 = (data['time'] >= s[ind + 1]) & (data['time'] <= e[ind + 1])
	data_trim = data[logical0 | logical1]
	len0 = sum(logical0)
	len1 = sum(logical1)
	ts_final = np.zeros((size, length, 3))
	label_final = [new_label] * size
	cp_final = np.zeros((size,))

	result = extract(len0, len1, length, size, ntrim=num_trim)
	cp_final = result['cp'].astype('int32')
	sample = result['sample'].astype('int32')
	for i in range(size):
		ts_temp = data_trim.iloc[sample[i, :], 1:4]
		ts_final[i, :, :] = ts_temp.to_numpy()

	return {"cp": cp_final, "ts": ts_final, "label": label_final}


# define the function for the subject level
def labelSubject(subject_path, length, size, num_trim=100):
	# get the csv files
	all_files = os.listdir(subject_path)
	csv_files = list(filter(lambda f: f.endswith('.csv'), all_files))
	csv_files = list(filter(lambda f: f.startswith('HASC'), csv_files))
	for ind, fname in enumerate(csv_files):
		print(ind)
		print(fname)
		fname_label = fname.replace('-acc.csv', '.label')
		fname_label = posixpath.join(subject_path, fname_label)
		fname = posixpath.join(subject_path, fname)
		# load the labels
		label_dataset = pd.read_csv(
			fname_label,
			comment="#",
			delimiter=",",
			names=['start', 'end', 'state']
		)
		num_consecutive_states = label_dataset.shape[0]
		# load the dataset
		dataset = pd.read_csv(
			fname, comment="#", delimiter=",", names=['time', 'x', 'y', 'z']
		)
		if num_consecutive_states < 2:
			warnings.warn(
				"The length of times series exceeds the minimum length of two segments. Reduce length or increase the num_trim.",
				DeprecationWarning,
			)

		for i in range(num_consecutive_states - 1):
			result = labelTransition(
				dataset, label_dataset, i, length, size, num_trim=num_trim
			)
			if i == 0:
				ts = result['ts']
				label = result['label']
				cp = result['cp']
			else:
				ts = np.concatenate([ts, result['ts']], axis=0)
				cp = np.concatenate([cp, result['cp']])
				label += result['label']

		if ind == 0:
			ts_ind = ts
			label_ind = label
			cp_ind = cp
		else:
			ts_ind = np.concatenate([ts_ind, ts], axis=0)
			cp_ind = np.concatenate([cp_ind, cp])
			label_ind += label

	return {"cp": cp_ind, "ts": ts_ind, "label": label_ind}


def extract(n1, n2, length, size, ntrim):
	n = n1 + n2
	ts = np.arange(n)
	if length > n - size:
		warnings.warn(
			"Not enough sample.",
			DeprecationWarning,
		)
	if n1 < ntrim + size or n2 < ntrim + size:
		warnings.warn(
			"One segment has not enough sample.",
			DeprecationWarning,
		)

	cp = np.zeros((size,))
	sample = np.zeros((size, length))
	len_half = length // 2
	if n1 <= n2:
		if n1 >= ntrim + size and n1 <= len_half:
			cp[0] = n1
			s = 0
			sample[0,] = ts[s:length]
			if size == 1:
				return {'cp': cp, 'sample': sample}
			else:
				s_set = np.random.choice(
					range(
						max(1, n1 + ntrim - length),
						min(n1 - ntrim, n - length)
					), (size - 1,),
					replace=False
				)
				for ind, s in enumerate(s_set):
					cp[ind + 1] = n1 - s
					sample[ind + 1, :] = ts[s:s + length]

				return {'cp': cp, 'sample': sample}
		elif n1 > len_half:
			cp[0] = len_half
			s = n1 - len_half
			sample[0,] = ts[n1 - len_half:n1 + len_half]
			if size == 1:
				return {'cp': cp, 'sample': sample}
			else:
				s_set = np.random.choice(
					range(
						max(0, n1 + ntrim - length),
						min(n1 - ntrim, n - length)
					), (size - 1,),
					replace=False
				)
				for ind, s in enumerate(s_set):
					cp[ind + 1] = n1 - s
					sample[ind + 1, :] = ts[s:s + length]

				return {'cp': cp, 'sample': sample}
	else:
		if n2 >= ntrim + size and n2 <= len_half:
			cp[0] = length - n2
			s = n - length
			sample[0,] = ts[s:n]
			if size == 1:
				return {'cp': cp, 'sample': sample}
			else:
				s_set = np.random.choice(
					range(
						max(0, n1 + ntrim - length),
						min(n1 - ntrim, n - 1 - length)
					), (size - 1,),
					replace=False
				)
				for ind, s in enumerate(s_set):
					cp[ind + 1] = n1 - s
					sample[ind + 1, :] = ts[s:s + length]

				return {'cp': cp, 'sample': sample}
		elif n2 > len_half:
			cp[0] = len_half
			s = n1 - len_half
			sample[0,] = ts[n1 - len_half:n1 + len_half]
			if size == 1:
				return {'cp': cp, 'sample': sample}
			else:
				s_set = np.random.choice(
					range(
						max(0, n1 + ntrim - length),
						min(n1 - ntrim, n - length)
					), (size - 1,),
					replace=False
				)
				for ind, s in enumerate(s_set):
					cp[ind + 1] = n1 - s
					sample[ind + 1, :] = ts[s:s + length]

				return {'cp': cp, 'sample': sample}


# functions for  extracting null time series
def tsExtract(data_trim, new_label, length, size, len0):
	ts_final = np.zeros((size, length, 3))
	label_final = [new_label] * size

	sample = np.sort(
		np.random.choice(range(0, len0 - length), (size,), replace=False)
	)
	for i in range(size):
		ts_temp = data_trim.iloc[sample[i]:sample[i] + length, 1:4]
		ts_final[i, :, :] = ts_temp.to_numpy()

	return {"ts": ts_final, "label": label_final}


def ExtractSubject(subject_path, length, size):
	# get the csv files
	all_files = os.listdir(subject_path)
	csv_files = list(filter(lambda f: f.endswith('.csv'), all_files))
	csv_files = list(filter(lambda f: f.startswith('HASC'), csv_files))
	for ind, fname in enumerate(csv_files):
		print(ind)
		print(fname)
		fname_label = fname.replace('-acc.csv', '.label')
		fname_label = posixpath.join(subject_path, fname_label)
		fname = posixpath.join(subject_path, fname)
		# load the labels
		label_dataset = pd.read_csv(
			fname_label,
			comment="#",
			delimiter=",",
			names=['start', 'end', 'state']
		)
		num_consecutive_states = label_dataset.shape[0]
		# load the dataset
		dataset = pd.read_csv(
			fname, comment="#", delimiter=",", names=['time', 'x', 'y', 'z']
		)
		ts = np.array([]).reshape(0, length, 3)
		label = []
		for i in range(num_consecutive_states):
			s = label_dataset['start'][i]
			e = label_dataset['end'][i]
			new_label = label_dataset['state'][i]
			logical = (dataset['time'] >= s) & (dataset['time'] <= e)
			data_trim = dataset[logical]
			len0 = sum(logical)
			if len0 <= length + size:
				continue
			else:
				result = tsExtract(data_trim, new_label, length, size, len0)
				ts = np.concatenate([ts, result['ts']], axis=0)
				label += result['label']

		if ind == 0:
			ts_ind = ts
			label_ind = label
		else:
			ts_ind = np.concatenate([ts_ind, ts], axis=0)
			label_ind += label

	return {"ts": ts_ind, "label": label_ind}


def DataGenAlternative(
	N_sub,
	B,
	mu_L,
	n,
	ARcoef=0.0,
	tau_bound=2,
	B_bound=[0.5, 1.5],
	type='Gaussian',
	scale=0.1,
	sigma=1.0
):
	"""This function genearates the simulation data from alternative model of change in mean.

	Parameters
	----------
	N_sub : int
		The sample size of simulation data.
	B : float
		The signal-to-noise ratio of parameter space.
	mu_L : float
		The single at the left of change point.
	n : int
		The length of time series.
	ARcoef : float, optional
		The autoregressive parameter of AR(p) model, by default 0.0
	tau_bound : int, optional
		The lower bound of change point, by default 2
	B_bound : list, optional
		The upper and lower bound scalars of signal-to-noise, by default [0.5, 1.5]
	type : str, optional
		The different models, by default 'Gaussian'. type="AR0" means AR(p) noise with autoregressive parameter 'ARcoef'; type="ARH" means Cauchy noise with scale parameter 'scale'; type="ARrho" means AR(p) noise with random autoregressive parameter 'scale';
	scale : float, optional
		The scale parameter of Cauchy distribution, by default 0.1
	sigma : float, optional
		The standard variance of normal distribution, by default 1.0

	Returns
	-------
	dict
		data: size (N_sub,n);
		tau_alt: size (N_sub,); the change points
		mu_R: size (N_sub,); the single at the right of change point
	"""
	tau_all = np.random.randint(low=tau_bound, high=n - tau_bound, size=N_sub)
	eta_all = tau_all / n
	mu_R_abs_lower = B / np.sqrt(eta_all * (1 - eta_all))
	# max_mu_R = np.max(mu_R_abs_lower)
	sign_all = np.random.choice([-1, 1], size=N_sub)
	mu_R_all = np.zeros((N_sub,))
	data = np.zeros((N_sub, n))
	for i in range(N_sub):
		mu_R = sign_all[i] * (
			mu_L - np.random.uniform(
				low=B_bound[0] * mu_R_abs_lower[i],
				high=B_bound[1] * mu_R_abs_lower[i],
				size=1
			)
		)
		mu_R_all[i] = mu_R
		mu = np.array([mu_L, mu_R], dtype=np.float32)
		if type == 'Gaussian':
			data[i, :] = GenDataMean(1, n, cp=tau_all[i], mu=mu, sigma=1)
		elif type == 'AR0':
			data[i, :] = GenDataMeanAR(
				1, n, cp=tau_all[i], mu=mu, sigma=1, coef=ARcoef
			)
		elif type == 'ARH':
			data[i, :] = GenDataMeanARH(
				1, n, cp=tau_all[i], mu=mu, coef=ARcoef, scale=scale
			)
		elif type == 'ARrho':
			data[i, :] = GenDataMeanARrho(
				1, n, cp=tau_all[i], mu=mu, sigma=sigma
			)

	return {"data": data, "tau_alt": tau_all, "mu_R_alt": mu_R_all}


def ComputeCUSUM(x):
	"""
		Compute the CUSUM with O(n) time complexity
	"""
	n = len(x)
	mean_left = x[0]
	mean_right = np.mean(x[1:])
	a = np.repeat(0.0, n - 1)
	a[0,] = np.sqrt((n - 1) / n) * (mean_left - mean_right)
	for i in range(1, n - 1):
		mean_left = mean_left + (x[i] - mean_left) / (i + 1)
		mean_right = mean_right + (mean_right - x[i]) / (n - i - 1)
		a[i,] = np.sqrt((n - i - 1) * (i + 1) / n) * (mean_left - mean_right)

	return a


def MaxCUSUM(x, T0=None):
	y = np.abs(ComputeCUSUM(x))
	if T0 is None:
		return np.max(y)
	else:
		return np.max(y[T0 - 1])
