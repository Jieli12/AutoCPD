"""
Author         : Jie Li, Department of Statistics, London School of Economics.
Date           : 2022-01-12 15:19:50
Last Author    : Jie Li
Last Revision  : 2022-10-23 10:34:21
File Path      : /AI-assisstedChangePointDetection/Python/utilsMultimode.py
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
	coef : float array
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
	coef : float array
		the coefficients of AR(p) model

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


def ConvertMean(data, window_size):
	"""
	Compute the window mean

	Parameters
	----------
	data : numpy array
		the 2-D array
	window_size : int
		the window size

	Returns
	-------
	numpy array
		2-D arrary
	"""
	cumsum = np.cumsum(data, axis=1)
	cumsum[:, window_size:] = cumsum[:, window_size:] - cumsum[:, :-window_size]
	return cumsum[:, window_size - 1:] / window_size


def ConvertVariance(data, window_size):
	"""
	Compute the window variance

	Parameters
	----------
	data : numpy array
		the 2-D array
	window_size : int
		the window size

	Returns
	-------
	numpy array
		2-D arrary
	"""
	n, p = data.shape
	p_new = p - window_size + 1
	variance = np.zeros((n, p_new))
	for i in range(p_new):
		variance[:, i] = np.var(data[:, i:i + window_size], axis=1)
	return variance


def ConvertSlope(data_y, data_x, window_size):
	"""
	Compute the window regression coefficients

	Parameters
	----------
	data_y : numpy array
		the 2-D array
	data_x : numpy array
		the 1-D array
	window_size : int
		the window size

	Returns
	-------
	numpy array
		list of 2-D arrary (constant, slope)
	"""
	n, p = data_y.shape
	p_new = p - window_size + 1
	intercept = slope = np.zeros((n, p_new))
	for i in range(p_new):
		x = data_x[i:i + window_size]
		for j in range(n):
			y = data_y[j, i:i + window_size]
			slope[j, i], intercept[j, i], _, _, _ = linregress(x, y)

	return intercept, slope


def Transform3(data_y, data_x, window_size, HF=False, bandwidth=5):
	"""
	Apply 3 transformations to the same dataset

	Parameters
	----------
	data_y : numpy array
		the 2-D array
	data_x : numpy array
		the 1-D array
	window_size : int
		the window size
	HF : logical
		the Haar-Fisz transformation if true, otherwise not
	bandwidth : int
		the bandwidth for isotonic regression in Haar-Fisz transformation

	Returns
	-------
	numpy array
		3-D arrary with size (N, 3,  n^{\prime})
	"""
	N, n = data_y.shape
	n_new = n - window_size + 1
	m = ConvertMean(data_y, window_size)
	if HF is True:
		v = HFTransformMatrix(data_y, bandwidth)
		window_half = window_size // 2
		v = v[:, window_half:-(window_size - window_half - 1)]
	else:
		v = ConvertVariance(data_y, window_size)

	_, s = ConvertSlope(data_y, data_x, window_size)
	m = m.reshape((N, 1, n_new))
	v = v.reshape((N, 1, n_new))
	s = s.reshape((N, 1, n_new))
	return np.concatenate((m, v, s), axis=1)


def Transform3New(data_y, diff=False):
	"""
	Apply 3 transformations (standardization, squaring, log transform) to the same dataset

	Parameters
	----------
	data_y : numpy array
		the 2-D array
	diff : logical
		if false, using the logarithm transformation otherwise using 1st order differentiation.

	Returns
	-------
	numpy array
		3-D arrary with size (N, 3,  n)
	"""
	N, n = data_y.shape
	m2 = np.square(data_y)
	m2log = np.log(m2)
	y_new = data_y.reshape((N, 1, n))
	m2 = m2.reshape((N, 1, n))
	m2log = m2log.reshape((N, 1, n))
	return np.concatenate((y_new, m2, m2log), axis=1)


def Transform4New(data_y):
	"""
	Apply 4 transformations (standardization, squaring, log transform, differentiation) to the same dataset

	Parameters
	----------
	data_y : numpy array
		the 2-D array

	Returns
	-------
	numpy array
		3-D arrary with size (N, 4,  n)
	"""
	N, n = data_y.shape
	m2 = np.square(data_y)
	m2log = np.log(m2)
	mdiff = np.diff(data_y)
	mdiff = np.concatenate((mdiff, mdiff[:, -1].reshape((N, 1))), axis=1)
	y_new = data_y.reshape((N, 1, n))
	m2 = m2.reshape((N, 1, n))
	m2log = m2log.reshape((N, 1, n))
	mdiff = mdiff.reshape((N, 1, n))
	return np.concatenate((y_new, m2, y_new, mdiff), axis=1)


def LogTransform(data):
	"""
	Log transformation

	Parameters
	----------
	data : float
		2-D array

	Returns
	-------
	float
		2-D array
	"""
	return np.log(np.square(data))


def HFTransform(x, bandwidth):
	"""
	The Haar-Fisz transformation. This routine is rewritten based on the function ddhf.trans in the software for the paper "Data-driven wavelet-Fisz methodology for nonparametric function estimation" by Piotr Fryzlewicz. See https://stats.lse.ac.uk/fryzlewicz/ddwf/ddwf.html

	Parameters
	----------
	x : float
		1-D array
	bandwidth : int
		bandwidth, the whole bandwidth is 2*bandwidth+1

	Returns
	-------
	float
		the Haar-Fisz transformation, 1-D array
	"""
	n = x.shape[0]
	n_half = np.int32(n / 2)
	J = np.int32(np.log2(n))
	hft = np.copy(x)
	sm = np.zeros((n_half,))
	det = np.zeros((n_half,))
	bw = [2 * bandwidth + 1]
	kr = KernelReg(
		endog=x, exog=np.arange(1, n + 1), var_type='c', reg_type='lc', bw=bw
	)
	alpha_hat = kr.fit(np.arange(1, n + 1))
	alpha_hat = alpha_hat[0]
	ind = np.argsort(alpha_hat)
	u = alpha_hat[ind]
	eps_hat_2 = np.square(x - alpha_hat)
	ir = IsotonicRegression(out_of_bounds="clip")
	h = ir.fit_transform(np.arange(1, n + 1), eps_hat_2[ind])

	for i in range(J):
		ind_even = 2 * np.arange(0, n_half)
		ind_odd = ind_even + 1
		sm[0:n_half] = (hft[ind_even] + hft[ind_odd]) / 2
		det[0:n_half] = (hft[ind_even] - hft[ind_odd]) / 2
		v = functionFromVector(u, np.sqrt(h), sm[0:n_half])
		det[v > 0] = det[v > 0] / v[v > 0]
		hft[:n_half] = sm[0:n_half]
		hft[n_half:n] = det[0:n_half]
		n = n // 2
		if np.mod(n, 2) == 1:
			n = n - 1
		n_half = n // 2
		sm = np.zeros((n_half,))
		det = np.zeros((n_half,))

	n_half = 1
	n = 2
	for i in range(J):
		sm = np.zeros((n_half,))
		det = np.zeros((n_half,))
		sm[0:n_half] = hft[0:n_half]
		det[0:n_half] = hft[n_half:n]
		hft[2 * np.arange(0, n_half)] = sm[0:n_half] + det[0:n_half]
		hft[(2 * np.arange(0, n_half)) + 1] = sm[0:n_half] - det[0:n_half]
		n_half = n
		n = 2 * n

	return hft


def functionFromVector(x, y, vec):
	"""
	Function from vector. This routine is rewritten based on the function function.from.vector in the software for the paper "Data-driven wavelet-Fisz methodology for nonparametric function estimation" by Piotr Fryzlewicz. See https://stats.lse.ac.uk/fryzlewicz/ddwf/ddwf.html.

	Parameters
	----------
	x : float
		1-D array
	y : float
		1-D array
	vec : float
		1-D array with half length

	Returns
	-------
	float
		1-D array with half length
	"""
	abs_diff = np.abs(np.expand_dims(vec, 1) - x)
	ind = np.argmin(abs_diff, axis=1)
	return y[ind]


def HFTransformMatrix(data, bandwidth):
	"""
	Apply HFTransform to the rows of the data matrix

	Parameters
	----------
	data : float
		2-D array
	bandwidth : int
		half bandwidth, the whole bandwidth is 2*bandwidth+1

	Returns
	-------
	float
		Haar-Fisz transform matrix
	"""
	data = np.square(data)
	n, p = data.shape
	x_trans = np.zeros((n, p))
	for i in range(n):
		x = data[i, :]
		x_trans[i, :] = HFTransform(x, bandwidth=bandwidth)
	return x_trans


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


def Transform2D0(data_y):
	"""
	Apply 4 transformations (original, squared, log squared, tanh) to the same dataset

	Parameters
	----------
	data_y : numpy array
		the 2-D array

	Returns
	-------
	numpy array
		3-D arrary with size (N, 4,  n)
	"""
	N, n = data_y.shape
	y_new = data_y.reshape((N, 1, n))
	return np.concatenate((y_new, y_new, y_new, y_new), axis=1)


def Transform2Dsq(data_y):
	"""
	Apply 4 transformations (original, squared, log squared, tanh) to the same dataset

	Parameters
	----------
	data_y : numpy array
		the 2-D array

	Returns
	-------
	numpy array
		3-D arrary with size (N, 4,  n)
	"""
	N, n = data_y.shape
	y_new = data_y.reshape((N, 1, n))
	m2 = np.square(data_y)
	m2 = m2.reshape((N, 1, n))
	return np.concatenate((y_new, m2, y_new, y_new), axis=1)


def Transform2Dsqlog(data_y):
	"""
	Apply 4 transformations (original, squared, log squared, tanh) to the same dataset

	Parameters
	----------
	data_y : numpy array
		the 2-D array

	Returns
	-------
	numpy array
		3-D arrary with size (N, 4,  n)
	"""
	N, n = data_y.shape
	data_y[data_y == 0.0] = 1e-8
	m2 = np.square(data_y)
	m2log = np.log(m2)

	y_new = data_y.reshape((N, 1, n))
	m2 = m2.reshape((N, 1, n))
	m2log = m2log.reshape((N, 1, n))

	return np.concatenate((y_new, m2, m2log, y_new), axis=1)


def Transform2Dtanh(data_y):
	"""
	Apply 4 transformations (original, squared, log squared, tanh) to the same dataset

	Parameters
	----------
	data_y : numpy array
		the 2-D array

	Returns
	-------
	numpy array
		3-D arrary with size (N, 4,  n)
	"""
	N, n = data_y.shape
	data_y[data_y == 0.0] = 1e-8
	m2 = np.square(data_y)
	m2log = np.log(m2)
	m_tanh = np.tanh(data_y)
	y_new = data_y.reshape((N, 1, n))
	m2 = m2.reshape((N, 1, n))
	m2log = m2log.reshape((N, 1, n))
	m_tanh = m_tanh.reshape((N, 1, n))
	return np.concatenate((y_new, m2, m_tanh, y_new), axis=1)


def Transform2DCUMSUM(data_y):
	"""
	Apply 4 transformations (original, squared, log squared, cumsum) to the same dataset

	Parameters
	----------
	data_y : numpy array
		the 2-D array

	Returns
	-------
	numpy array
		3-D arrary with size (N, 4,  n)
	"""
	N, n = data_y.shape
	data_y[data_y == 0.0] = 1e-8
	m2 = np.square(data_y)
	m2log = np.log(m2)
	m_cumsum = np.cumsum(data_y)
	y_new = data_y.reshape((N, 1, n))
	m2 = m2.reshape((N, 1, n))
	m2log = m2log.reshape((N, 1, n))
	m_cumsum = m_cumsum.reshape((N, 1, n))
	return np.concatenate((y_new, m2, m2log, m_cumsum), axis=1)


def Transform2Dsqlogtanh(data_y):
	"""
	Apply 4 transformations (original, squared, log squared, tanh) to the same dataset

	Parameters
	----------
	data_y : numpy array
		the 2-D array

	Returns
	-------
	numpy array
		3-D arrary with size (N, 4,  n)
	"""
	N, n = data_y.shape
	data_y[data_y == 0.0] = 1e-8
	m2 = np.square(data_y)
	m2log = np.log(m2)
	m_tanh = np.tanh(data_y)
	y_new = data_y.reshape((N, 1, n))
	m2 = m2.reshape((N, 1, n))
	m2log = m2log.reshape((N, 1, n))
	m_tanh = m_tanh.reshape((N, 1, n))
	return np.concatenate((y_new, m2log, m_tanh, y_new), axis=1)


def Transform2Dsqlogonly(data_y):
	"""
	Apply 4 transformations (original, squared, log squared, tanh) to the same dataset

	Parameters
	----------
	data_y : numpy array
		the 2-D array

	Returns
	-------
	numpy array
		3-D arrary with size (N, 4,  n)
	"""
	N, n = data_y.shape
	data_y[data_y == 0.0] = 1e-8
	m2 = np.square(data_y)
	m2log = np.log(m2)
	m_tanh = np.tanh(data_y)
	y_new = data_y.reshape((N, 1, n))
	m2 = m2.reshape((N, 1, n))
	m2log = m2log.reshape((N, 1, n))
	m_tanh = m_tanh.reshape((N, 1, n))
	return np.concatenate((y_new, m2log, y_new, y_new), axis=1)


def Transform2Dtanhonly(data_y):
	"""
	Apply 4 transformations (original, squared, log squared, tanh) to the same dataset

	Parameters
	----------
	data_y : numpy array
		the 2-D array

	Returns
	-------
	numpy array
		3-D arrary with size (N, 4,  n)
	"""
	N, n = data_y.shape
	data_y[data_y == 0.0] = 1e-8
	m2 = np.square(data_y)
	m2log = np.log(m2)
	m_tanh = np.tanh(data_y)
	y_new = data_y.reshape((N, 1, n))
	m2 = m2.reshape((N, 1, n))
	m2log = m2log.reshape((N, 1, n))
	m_tanh = m_tanh.reshape((N, 1, n))
	return np.concatenate((y_new, m_tanh, y_new, y_new), axis=1)


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
		# features_to_normalize = ['x', 'y', 'z']
		# dataset[features_to_normalize] = dataset[features_to_normalize].apply(
		# 	lambda x: 2 * (x - x.min()) / (x.max() - x.min()) - 1
		# )
		# define the variables for each csv file
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
		# features_to_normalize = ['x', 'y', 'z']
		# dataset[features_to_normalize] = dataset[features_to_normalize].apply(
		# 	lambda x: 2 * (x - x.min()) / (x.max() - x.min()) - 1
		# )
		ts = np.array([]).reshape(0, length, 3)
		# ts = np.empty((1, length, 3))
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
	"""
	Apply 4 transformations (original, squared, log squared, tanh) to the same dataset

	Parameters
	----------
	data_y : numpy array
		the 2-D array

	Returns
	-------
	numpy array
		3-D arrary with size (N, 4,  n)
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


def DataGenAlternativeAR(N_sub, B, mu_L, n, ARcoef, tau_bound=50, coef=1.5):
	"""
	Apply 4 transformations (original, squared, log squared, tanh) to the same dataset

	Parameters
	----------
	data_y : numpy array
		the 2-D array

	Returns
	-------
	numpy array
		3-D arrary with size (N, 4,  n)
	"""
	tau_all = np.random.randint(low=tau_bound, high=n - tau_bound, size=N_sub)
	eta_all = tau_all / n
	mu_R_abs_lower = B / np.sqrt(eta_all * (1 - eta_all))
	max_mu_R = np.max(mu_R_abs_lower)
	sign_all = np.random.choice([-1, 1], size=N_sub)
	mu_R_all = np.zeros((N_sub,))
	data = np.zeros((N_sub, n))
	for i in range(N_sub):
		mu_R = sign_all[i] * (
			mu_L - np.random.uniform(
				low=mu_R_abs_lower[i], high=coef * mu_R_abs_lower[i], size=1
			)
		)
		mu_R_all[i] = mu_R
		mu = np.array([mu_L, mu_R], dtype=np.float32)
		data[i, :] = GenDataMeanAR(
			1, n, cp=tau_all[i], mu=mu, sigma=1, coef=ARcoef
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
