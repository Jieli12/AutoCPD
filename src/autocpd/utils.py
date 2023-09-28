import os
import posixpath
import warnings
from itertools import groupby

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.special import gamma
from scipy.stats import cauchy, rankdata
from sklearn.utils import shuffle
from statsmodels.tsa.arima_process import ArmaProcess


def GenDataMean(N, n, cp, mu, sigma):
    """
    The function  generates the data for change in mean with Gaussian noise.
    When "cp" is None, it generates the data without change point.

    Parameters
    ----------
    N : int
        the sample size
    n : int
        the length of time series
    cp : int
        the change point, only 1 change point is accepted in this function.
    mu : float
        the piecewise mean
    sigma : float
        the standard deviation of Gaussian distribution

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
    The function  generates the data for change in mean with AR(1) noise.
    When "cp" is None, it generates the data without change point.

    Parameters
    ----------
    N : int
        the sample size
    n : int
        the length of time series
    cp : int
        the change point, only 1 change point is accepted in this function.
    mu : float
        the piecewise mean
    sigma : float
        the standard deviation of Gaussian innovations in AR(1) noise
    coef : float scalar
        the coefficients of AR(1) model

    Returns
    -------
    numpy array
        2D array with size (N, n)
    """
    arparams = np.array([1, -coef])
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
    The function  generates the data for change in mean + Cauchy noise with
    location parameter 0 and scale parameter 'scale'. When "cp" is None, it
    generates the data without change point.

    Parameters
    ----------
    N : int
        the sample size
    n : int
        the length of time series
    cp : int
        the change point, only 1 change point is accepted in this function.
    mu : float
        the piecewise mean
    coef : float array
        the coefficients of AR(1) model
    scale : the scale parameter of Cauchy distribution
        the coefficients of AR(1) model

    Returns
    -------
    numpy array
        2D array with size (N, n)
    """
    # initialize
    n1 = n + 30
    x_0 = np.ones((N,), dtype=np.float64)
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
    The function  generates the data for change in mean with AR(1) noise. The
    autoregressive coefficient is generated from standard uniform distribution.
    When "cp" is None, it generates the data without change point.

    Parameters
    ----------
    N : int
        the sample size
    n : int
        the length of time series
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
    The function  generates the data for change in variance with piecewise
    constant signal. When "cp" is None, it generates the data without change
    point in variance.

    Parameters
    ----------
    N : int
        the sample size
    n : int
        the length of time series
    cp : int
        the change point, only 1 change point is accepted in this function.
    mu : float
        the piecewise mean
    sigma : float
        the standard deviation of Gaussian distribution

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
    The function  generates the data for change in slope with Gaussian noise.
    When "cp" is None, it generates the data without change point in slope.

    Parameters
    ----------
    N : int
        the sample size
    n : int
        the length of time series
    cp : int
        the change point, only 1 change point is accepted in this function.
    slopes : float
        the slopes before and after the change point
    sigma : float
        the standard deviation of Gaussian distribution
    start : float
        the y-intercept of linear model

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
    """Apply 4 transformations (original, squared, log squared, tanh) to the same dataset

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
        data_y = (data_y - data_y.min(axis=0)) / (
            data_y.max(axis=0) - data_y.min(axis=0)
        )
        if cumsum is True:
            m_tanh = np.cumsum(data_y)
        else:
            m_tanh = np.tanh(data_y)

        m2 = (m2 - m2.min(axis=0)) / (m2.max(axis=0) - m2.min(axis=0))
        m2log = (m2log - m2log.min(axis=0)) / (m2log.max(axis=0) - m2log.min(axis=0))
        m_tanh = (m_tanh - m_tanh.min(axis=0)) / (
            m_tanh.max(axis=0) - m_tanh.min(axis=0)
        )
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

    y_new = data_y.reshape((N, 1, n))
    m2 = m2.reshape((N, 1, n))
    m2log = m2log.reshape((N, 1, n))
    m_tanh = m_tanh.reshape((N, 1, n))
    return np.concatenate((y_new, m2, m2log, m_tanh), axis=1)


def labelTransition(data, label, ind, length, size, num_trim=100):
    """get the transition labels, change-points and time series from one subject

    Parameters
    ----------
    data : DataFrame
        the time series.
    label : DataFrame
        the states of the subject
    ind : scalar
        the index of state
    length : int
        the length of extracted time series
    size : int
        the sample size
    num_trim : int, optional
        the number of observations to be trimmed before and after the change-point, by default 100

    Returns
    -------
    dictionary
        cp: the change-points; ts: time series; label: the transition labels.
    """
    s = label["start"][ind : ind + 2]
    e = label["end"][ind : ind + 2]
    state = label["state"][ind : ind + 2]
    new_label = state[ind] + "->" + state[ind + 1]
    logical0 = (data["time"] >= s[ind]) & (data["time"] <= e[ind])
    logical1 = (data["time"] >= s[ind + 1]) & (data["time"] <= e[ind + 1])
    data_trim = data[logical0 | logical1]
    len0 = sum(logical0)
    len1 = sum(logical1)
    ts_final = np.zeros((size, length, 3))
    label_final = [new_label] * size
    cp_final = np.zeros((size,))

    result = extract(len0, len1, length, size, ntrim=num_trim)
    cp_final = result["cp"].astype("int32")
    sample = result["sample"].astype("int32")
    for i in range(size):
        ts_temp = data_trim.iloc[sample[i, :], 1:4]
        ts_final[i, :, :] = ts_temp.to_numpy()

    return {"cp": cp_final, "ts": ts_final, "label": label_final}


def labelSubject(subject_path, length, size, num_trim=100):
    """
    obtain the transition labels, change-points and time series from one subject.

    Parameters
    ----------
    subject_path : string
        the path of subject data
    length : int
        the length of extracted time series
    size : int
        the sample size
    num_trim : int, optional
        the number of observations to be trimmed before and after the change-point, by default 100

    Returns
    -------
    dictionary
        cp: the change-points; ts: time series; label: the transition labels.
    """
    # get the csv files
    all_files = os.listdir(subject_path)
    csv_files = list(filter(lambda f: f.endswith(".csv"), all_files))
    csv_files = list(filter(lambda f: f.startswith("HASC"), csv_files))
    for ind, fname in enumerate(csv_files):
        print(ind)
        print(fname)
        fname_label = fname.replace("-acc.csv", ".label")
        fname_label = posixpath.join(subject_path, fname_label)
        fname = posixpath.join(subject_path, fname)
        # load the labels
        label_dataset = pd.read_csv(
            fname_label, comment="#", delimiter=",", names=["start", "end", "state"]
        )
        num_consecutive_states = label_dataset.shape[0]
        # load the dataset
        dataset = pd.read_csv(
            fname, comment="#", delimiter=",", names=["time", "x", "y", "z"]
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
                ts = result["ts"]
                label = result["label"]
                cp = result["cp"]
            else:
                ts = np.concatenate([ts, result["ts"]], axis=0)
                cp = np.concatenate([cp, result["cp"]])
                label += result["label"]

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
    """
    This function randomly extracts samples (consecutive segments) with
    length 'length' from a time series concatenated by two different time
    series with length 'n1' and 'n2' respectively. Argument 'ntrim' controls
    the minimum distance between change-point and start or end point of
    consecutive segment. It returns a dictionary containing two arrays: cp and
    sample. cp is an array of change points. sample is a 2D array where each
    row is the indices of consecutive segment .


    Parameters
    ----------
    n1 : the length of signal before change-point
        _description_
    n2 : int
        the length of time series after change-point
    length : int
        the length of time series segment that we want to extract
    size : int
        the sample size
    ntrim : int
        the number of observations to be trimmed before and after the change-point

    Returns
    -------
    dict
        'cp' is the set of change-points. 'sample' is a matrix of indices
    """
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
                return {"cp": cp, "sample": sample}

            s_set = np.random.choice(
                range(max(1, n1 + ntrim - length), min(n1 - ntrim, n - length)),
                (size - 1,),
                replace=False,
            )
            for ind, s in enumerate(s_set):
                cp[ind + 1] = n1 - s
                sample[ind + 1, :] = ts[s : s + length]

            return {"cp": cp, "sample": sample}
        if n1 > len_half:
            cp[0] = len_half
            s = n1 - len_half
            sample[0,] = ts[n1 - len_half : n1 + len_half]
            if size == 1:
                return {"cp": cp, "sample": sample}
            s_set = np.random.choice(
                range(max(0, n1 + ntrim - length), min(n1 - ntrim, n - length)),
                (size - 1,),
                replace=False,
            )
            for ind, s in enumerate(s_set):
                cp[ind + 1] = n1 - s
                sample[ind + 1, :] = ts[s : s + length]

            return {"cp": cp, "sample": sample}
    else:
        if n2 >= ntrim + size and n2 <= len_half:
            cp[0] = length - n2
            s = n - length
            sample[0,] = ts[s:n]
            if size == 1:
                return {"cp": cp, "sample": sample}
            s_set = np.random.choice(
                range(max(0, n1 + ntrim - length), min(n1 - ntrim, n - 1 - length)),
                (size - 1,),
                replace=False,
            )
            for ind, s in enumerate(s_set):
                cp[ind + 1] = n1 - s
                sample[ind + 1, :] = ts[s : s + length]

            return {"cp": cp, "sample": sample}
        if n2 > len_half:
            cp[0] = len_half
            s = n1 - len_half
            sample[0,] = ts[n1 - len_half : n1 + len_half]
            if size == 1:
                return {"cp": cp, "sample": sample}
            s_set = np.random.choice(
                range(max(0, n1 + ntrim - length), min(n1 - ntrim, n - length)),
                (size - 1,),
                replace=False,
            )
            for ind, s in enumerate(s_set):
                cp[ind + 1] = n1 - s
                sample[ind + 1, :] = ts[s : s + length]

            return {"cp": cp, "sample": sample}


# functions for  extracting null time series
def tsExtract(data_trim, new_label, length, size, len0):
    """
    To extract the labels without change-points

    Parameters
    ----------
    data_trim : DataFrame
        the dataset of one specific state
    new_label : DataFrame
        the label, not transition label.
    length : int
        the length of extracted time series
    size : int
        the sample size
    len0 : int
        the length of time series for one specific state

    Returns
    -------
    dict
        ts: time series; label: the labels.
    """
    ts_final = np.zeros((size, length, 3))
    label_final = [new_label] * size

    sample = np.sort(np.random.choice(range(0, len0 - length), (size,), replace=False))
    for i in range(size):
        ts_temp = data_trim.iloc[sample[i] : sample[i] + length, 1:4]
        ts_final[i, :, :] = ts_temp.to_numpy()

    return {"ts": ts_final, "label": label_final}


def ExtractSubject(subject_path, length, size):
    """
    To extract the null labels without change-points from one subject

    Parameters
    ----------
    subject_path : string
        the path of subject data
    length : int
        the length of extracted time series
    size : int
        the sample size

    Returns
    -------
    dict
        ts: time series; label: the labels.
    """
    # get the csv files
    all_files = os.listdir(subject_path)
    csv_files = list(filter(lambda f: f.endswith(".csv"), all_files))
    csv_files = list(filter(lambda f: f.startswith("HASC"), csv_files))
    for ind, fname in enumerate(csv_files):
        print(ind)
        print(fname)
        fname_label = fname.replace("-acc.csv", ".label")
        fname_label = posixpath.join(subject_path, fname_label)
        fname = posixpath.join(subject_path, fname)
        # load the labels
        label_dataset = pd.read_csv(
            fname_label, comment="#", delimiter=",", names=["start", "end", "state"]
        )
        num_consecutive_states = label_dataset.shape[0]
        # load the dataset
        dataset = pd.read_csv(
            fname, comment="#", delimiter=",", names=["time", "x", "y", "z"]
        )
        ts = np.array([]).reshape((0, length, 3))
        label = []
        for i in range(num_consecutive_states):
            s = label_dataset["start"][i]
            e = label_dataset["end"][i]
            new_label = label_dataset["state"][i]
            logical = (dataset["time"] >= s) & (dataset["time"] <= e)
            data_trim = dataset[logical]
            len0 = sum(logical)
            if len0 <= length + size:
                continue
            else:
                result = tsExtract(data_trim, new_label, length, size, len0)
                ts = np.concatenate([ts, result["ts"]], axis=0)
                label += result["label"]

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
    B_bound,
    ARcoef=0.0,
    tau_bound=2,
    ar_model="Gaussian",
    scale=0.1,
    sigma=1.0,
):
    """
    This function genearates the simulation data from alternative model of change in mean.

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
    B_bound : list, optional
        The upper and lower bound scalars of signal-to-noise.
    ARcoef : float, optional
        The autoregressive parameter of AR(1) model, by default 0.0
    tau_bound : int, optional
        The lower bound of change point, by default 2
    ar_model : str, optional
        The different models, by default 'Gaussian'. ar_model="AR0" means AR(1)
        noise with autoregressive parameter 'ARcoef'; ar_model="ARH" means
        Cauchy noise with scale parameter 'scale'; ar_model="ARrho" means AR(1)
        noise with random autoregressive parameter 'scale';
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
        mu_R = mu_L - sign_all[i] * np.random.uniform(
            low=B_bound[0] * mu_R_abs_lower[i],
            high=B_bound[1] * mu_R_abs_lower[i],
            size=1,
        )
        mu_R_all[i] = mu_R[0]
        mu = np.array([mu_L, mu_R[0]], dtype=np.float32)
        if ar_model == "Gaussian":
            data[i, :] = GenDataMean(1, n, cp=tau_all[i], mu=mu, sigma=1)
        elif ar_model == "AR0":
            data[i, :] = GenDataMeanAR(1, n, cp=tau_all[i], mu=mu, sigma=1, coef=ARcoef)
        elif ar_model == "ARH":
            data[i, :] = GenDataMeanARH(
                1, n, cp=tau_all[i], mu=mu, coef=ARcoef, scale=scale
            )
        elif ar_model == "ARrho":
            data[i, :] = GenDataMeanARrho(1, n, cp=tau_all[i], mu=mu, sigma=sigma)

    return {"data": data, "tau_alt": tau_all, "mu_R_alt": mu_R_all}


def ComputeCUSUM(x):
    """
    Compute the CUSUM statistics with O(n) time complexity

    Parameters
    ----------
    x : vector
        the time series

    Returns
    -------
    vector
        a: the CUSUM statistics vector.
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


def MaxCUSUM(x):
    """
    To return the maximum of CUSUM

    Parameters
    ----------
    x : vector
        the time series

    Returns
    -------
    scalar
        the maximum of CUSUM
    """
    y = np.abs(ComputeCUSUM(x))
    return np.max(y)


def Transform2D2TR(data_y, rescale=False, times=2):
    """Apply 2 transformations (original, squared) to the same dataset, each
    transformation is repeated user-specified times.

    Parameters
    ----------
    data_y : numpy array
        the 2-D array
    rescale : logical bool
        default False
    times : integer
        the number of repetitions

    Returns
    -------
    numpy array
        3-D arrary with size (N, 2*times,  n)
    """
    N, n = data_y.shape
    if rescale is True:
        data_y = (data_y - data_y.min(axis=1, keepdims=True)) / (
            data_y.max(axis=1, keepdims=True) - data_y.min(axis=1, keepdims=True)
        )
    data_y[data_y == 0.0] = 1e-8
    m2 = np.square(data_y)
    m2[m2 == 0.0] = 1e-8
    y_new = data_y.reshape((N, 1, n))
    m2 = m2.reshape((N, 1, n))
    y_new = np.repeat(y_new, times, axis=1)
    m2 = np.repeat(m2, times, axis=1)
    return np.concatenate((y_new, m2), axis=1)


def ComputeMeanVarNorm(x, minseglen=2):
    """
    Compute the likelihood for change in variance. Rewritten by the R function
    single.var.norm.calc() in package changepoint.

    Parameters
    ----------
    x : numpy array
        the time series
    minseglen : int
        the minimum length of segment

    Returns
    -------
    scalar
        the likelihood ratio
    """
    n = len(x)
    y = np.cumsum(x)
    y2 = np.cumsum(x**2)
    y = np.insert(y, 0, 0)
    y2 = np.insert(y2, 0, 0)
    null = n * np.log((y2[n] - y[n] ** 2 / n) / n)
    taustar = np.arange(minseglen, n - minseglen + 2)
    sigma1 = (y2[taustar] - y[taustar] ** 2 / taustar) / (taustar)
    neg = sigma1 <= 0
    sigma1[neg] = 1e-10
    sigman = ((y2[n] - y2[taustar]) - (y[n] - y[taustar]) ** 2 / (n - taustar)) / (
        n - taustar
    )
    neg = sigman <= 0
    sigman[neg] = 1e-10
    tmp = null - taustar * np.log(sigma1) - (n - taustar) * np.log(sigman)

    return np.sqrt(np.max(tmp))


def get_wilcoxon_test(x):
    """Compute the Wilcoxon statistics

    Parameters
    ----------
    x : array
        the time series

    Returns
    -------
    scalar
        the maximum Wilcoxon statistics
    """
    y = wilcoxon(x) / np.sqrt(get_asyvar_window(x))
    return np.max(y)


def wilcoxon(x):
    """This function implements the Wilcoxon cumulative sum statistic (Dehling
    et al, 2013, Eq (20)) for nonparametric change point detection.
    The following code is translated from the C function "wilcoxsukz" in R
    package "robts". The accuracy of this function is already been tested.

    Parameters
    ----------
    x : array
        time series

    Returns
    -------
    1D array
        the test statistic for each potential change point.
    """
    n = len(x)
    tn = np.repeat(0.0, n - 1)
    for k in range(1, n):
        tn_temp = 0
        for i in range(0, k):
            tn_temp = tn_temp + np.sum(x[k:n] > x[i]) - (n - k) / 2.0
        tn[k - 1] = tn_temp * np.sqrt(k * (n - k)) / n * 2
    return np.abs(tn / n ** (3 / 2))


def get_asyvar_window(x, momentp=1):
    """This function computes the asymptotic variance of long run dependence
    time series using "window" method. This function is translated from the R
    function "asymvar.window". This function is already been tested by letting
    "overlapping=F","obs="ranks".

    Parameters
    ----------
    x : 1D array
        The time series
    momentp : int, optional
        which centred mean should be used, see Peligrad and Shao (1995) for
        details, by default 1

    Returns
    -------
    scalar
        The asymptotic variance of time series.
    """
    n = len(x)
    x = rankdata(x) / n
    l = np.int32(np.round((3 * n) ** (1 / 3) + 1))
    phibar = np.mean(x)
    k = np.int32(n // l)
    xma = np.reshape(x[0 : (k * l)], (k, l), order="c")
    s = np.sum(xma, axis=1)
    s = (np.abs(s - l * phibar) / np.sqrt(l)) ** momentp
    cp = 2 ** (-momentp / 2) * np.sqrt(np.pi) / gamma((momentp + 1) / 2)
    er = np.sum(s) / k * cp
    return er ** (2 / momentp)


def DataGenScenarios(scenario, N, B, mu_L, n, B_bound, rho, tau_bound):
    """This function generates the data based  on  Scenarios 1, a and 3 in "Automatic Change-point Detection in Time Series via Deep Learning" (Jie et al. ,2023)

    Parameters
    ----------
    scenario : string
        the scenario label: 'A0' is the Scenarios 1 with 'rho=0', 'A07' is the Scenarios 1 with  'rho=0.7',  'C' is the Scenarios 2 and 'D' is the Scenarios 3 with heavy tailed noise.
    N : int
        the sample size
    B : float
        The signal-to-noise ratio of parameter space.
    mu_L : float
        The single at the left of change point.
    n : int
        The length of time series.
    B_bound : list, optional
        The upper and lower bound scalars of signal-to-noise.
    rho : scalar
        the autocorrelation of AR(1) model
    tau_bound : int, optional
        The lower bound of change point, by default 2

    Returns
    -------
    dict
        data_all: the time series; y_all: the label array.
    """
    np.random.seed(2022)  # numpy seed fixing
    if scenario == "A0":
        result = DataGenAlternative(
            N_sub=N,
            B=B,
            mu_L=mu_L,
            n=n,
            B_bound=B_bound,
            ARcoef=rho,
            tau_bound=tau_bound,
            ar_model="Gaussian",
        )
        data_alt = result["data"]
        #  generate dataset for null hypothesis
        data_null = GenDataMean(N, n, cp=None, mu=(mu_L, mu_L), sigma=1)
        data_all = np.concatenate((data_alt, data_null), axis=0)
        y_all = np.repeat((1, 0), N).reshape((2 * N, 1))
        #  generate the training dataset and test dataset
        data_all, y_all = shuffle(data_all, y_all, random_state=42)
    elif scenario == "A07":
        rho = 0.7
        result = DataGenAlternative(
            N_sub=N,
            B=B,
            mu_L=mu_L,
            n=n,
            B_bound=B_bound,
            ARcoef=rho,
            tau_bound=tau_bound,
            ar_model="AR0",
        )
        data_alt = result["data"]
        #  generate dataset for null hypothesis
        data_null = GenDataMeanAR(N, n, cp=None, mu=(mu_L, mu_L), sigma=1, coef=rho)
        data_all = np.concatenate((data_alt, data_null), axis=0)
        y_all = np.repeat((1, 0), N).reshape((2 * N, 1))
        #  generate the training dataset and test dataset
        data_all, y_all = shuffle(data_all, y_all, random_state=42)
    elif scenario == "C":
        scale = 0.3
        result = DataGenAlternative(
            N_sub=N,
            B=B,
            mu_L=mu_L,
            n=n,
            B_bound=B_bound,
            ARcoef=rho,
            tau_bound=tau_bound,
            ar_model="ARH",
            scale=scale,
        )
        data_alt = result["data"]
        #  generate dataset for null hypothesis
        data_null = GenDataMeanARH(
            N, n, cp=None, mu=(mu_L, mu_L), coef=rho, scale=scale
        )
        data_all = np.concatenate((data_alt, data_null), axis=0)
        y_all = np.repeat((1, 0), N).reshape((2 * N, 1))
        #  generate the training dataset and test dataset
        data_all, y_all = shuffle(data_all, y_all, random_state=42)
    elif scenario == "D":
        sigma = np.sqrt(2)
        result = DataGenAlternative(
            N_sub=N,
            B=B,
            mu_L=mu_L,
            n=n,
            B_bound=B_bound,
            tau_bound=tau_bound,
            ar_model="ARrho",
            sigma=sigma,
        )
        data_alt = result["data"]
        #  generate dataset for null hypothesis
        data_null = GenDataMeanARrho(N, n, cp=None, mu=(mu_L, mu_L), sigma=sigma)
        data_all = np.concatenate((data_alt, data_null), axis=0)
        y_all = np.repeat((1, 0), N).reshape((2 * N, 1))
        #  generate the training dataset and test dataset
        data_all, y_all = shuffle(data_all, y_all, random_state=42)
    return data_all, y_all


def GenerateARAll(N, n, coef_left, coef_right, sigma, tau_bound):
    """
    This function generates N the AR(1) signal

    Parameters
    ----------
    N : integer
        The number of observations
    n : integer
        _description_
    coef_left : float
        The AR coefficient before the change-point
    coef_right : float
        The AR coefficient after the change-point
    sigma : float
        The standard deviation of noise
    tau_bound : integer
        The bound of change-point

    Returns
    -------
    2D arrary and change-points
        dataset with size (2*N, n), N change-points
    """
    tau = np.random.randint(low=tau_bound, high=n - tau_bound, size=N)
    data = np.zeros((2 * N, n))
    for i in range(N):
        data[i, :] = GenerateAR(n, coef_left, coef_right, None, sigma)
        data[i + N, :] = GenerateAR(n, coef_left, coef_right, tau[i], sigma)

    return data, tau


def GenerateAR(n, coef_left, coef_right, tau, sigma):
    """This function generates the signal of AR(1) model

    Parameters
    ----------
    n : integer
        The length of time series
    coef_left : float
        The AR coefficient before the change-point
    coef_right : float
        The AR coefficient after the change-point
    tau : integer
        The location of change-point
    sigma : float
        The standard deviation of noise

    Returns
    -------
    array
        The time series with length n.
    """
    x = np.zeros((n + 1,))
    if tau is None:
        for i in range(n):
            x[i + 1] = coef_left * x[i] + np.random.normal(0, sigma, 1)
    else:
        for i in range(n):
            if i < tau:
                x[i + 1] = coef_left * x[i] + np.random.normal(0, sigma, 1)
            else:
                x[i + 1] = coef_right * x[i] + np.random.normal(0, sigma, 1)
    return x[1:]


def get_cusum_location(x):
    """
    This function return the estimation of change-point location based on CUSUM.

    Parameters
    ----------
    x : numpy array
        The time series

    Returns
    -------
    int
        change-point location
    """
    y = np.abs(ComputeCUSUM(x))
    return np.argmax(y) + 1


def ComputeMosum(x, G):
    """
    Compute the mosum statistic, rewritten according to mosum.stat function in
    mosum R package.

    Parameters
    ----------
    x : numpy array
        The time series
    G : scalar
        the width of moving window

    Returns
    -------
    int
        the location of maximum mosum statistics
    """
    n = len(x)
    G = int(G)
    sums_left = np.convolve(x, np.ones((G,)), "valid")
    sums_right = sums_left
    unscaled0 = np.repeat(np.nan, G - 1)
    unscaled2 = np.repeat(np.nan, G)
    unscaled1 = (sums_right[G:] - sums_left[0:-G]) / np.sqrt(2 * G)
    unscaledStatistic = np.concatenate((unscaled0, unscaled1, unscaled2))
    # MOSUM-based variance estimators
    summedSquares_left = np.convolve(x**2, np.ones((G,)), "valid")
    squaredSums_left = sums_left**2
    var_tmp_left = summedSquares_left - 1 / G * squaredSums_left
    var_left = np.concatenate((unscaled0, var_tmp_left)) / G
    var_tmp_right = var_tmp_left
    var_right = np.concatenate((var_tmp_right[1:], unscaled2)) / G
    var = (var_right + var_left) / 2
    weight_left = np.sqrt(2 * G / np.arange(1, G + 1) / np.arange(2 * G - 1, G - 1, -1))
    unscaledStatistic[0:G] = np.cumsum(np.mean(x[0 : 2 * G]) - x[0:G]) * weight_left
    var[0:G] = var[G - 1]
    weight_right = np.sqrt(2 * G / np.arange(G - 1, 0, -1) / np.arange(G + 1, 2 * G))
    x_rev = x[-2 * G :]
    unscaledStatistic[n - G : n - 1] = (
        np.cumsum(np.mean(x_rev) - x_rev)[-G:-1] * weight_right
    )
    unscaledStatistic[n - 1] = 0
    var[-G:] = var[n - G - 1]
    res = np.abs(unscaledStatistic) / np.sqrt(var)
    return np.argmax(res) + 1


def get_loc_3(model, x_test, n, width):
    """
    This function obtains locations of methods: NN, double mosum based on
    predicted label and probabilities.

    Parameters
    ----------
    model : model
        The trained model
    x_test : vector
        The vector of time series
    n : int
        The  length of x_test
    width : int
        The width of second moving window.

    Returns
    -------
    array
        3 locations.
    """
    pred, prob = get_label(model, x_test, n)
    loc_nn = get_mosum_loc_nn(pred, n)
    loc_label = get_mosum_loc_double(pred, n, width, False)
    loc_prob = get_mosum_loc_double(prob, n, width, True)
    return np.array([loc_nn, loc_label, loc_prob])


def get_label(model, x_test, n):
    """
    This function gets the predicted label for the testing time series:x_test

    Parameters
    ----------
    model : tensorflow model
        The trained tensorflow model
    x_test : vector
        The vector of time series
    n : int
        The width of moving window

    Returns
    -------
    arrays
        two arrays, one is predicted label, the other is probabilities.
    """
    num_sliding_windows = len(x_test) - n + 1
    x_test_mat = np.zeros((num_sliding_windows, n))
    for i in range(num_sliding_windows):
        x_test_mat[i, :] = x_test[i : i + n]

    y_prob = model.predict(x_test_mat)
    y_pred = np.argmax(y_prob, axis=1)
    return y_pred, y_prob


def get_mosum_loc_nn(pred, n):
    """This function return the estimation of change-point based on MOSUM using NN.

    Parameters
    ----------
    pred : vector
        The vector of predicted labels
    n : int
        The width of moving window

    Returns
    -------
    int
        change-point location
    """
    num_each_group = []
    state = []
    # get the states of consecutive groups and their length.
    for _, group in groupby(pred):
        g = list(group)
        state.append(g[0])
        num_each_group.append(len(g))

    state = np.asarray(state, dtype=np.int32)
    num_each_group = np.asarray(num_each_group, dtype=np.int32)
    # get the interval of longest consecutive 1 (alarm)
    index = state == 1
    num_group = len(num_each_group)
    aind = np.arange(num_group)[index]
    max_ind = np.argmax(num_each_group[index])
    interval = np.cumsum(num_each_group)
    a = interval[aind[max_ind] - 1] + 1
    b = interval[aind[max_ind]]
    return int(np.round((a + b + n) / 2))


def get_mosum_loc_double(x, n, width, use_prob):
    """This function return the estimation of change-point based on MOSUM by second moving average.

    Parameters
    ----------
    x : array
        either the predicted labels or probabilities
    n : int
        The width of moving window

    Returns
    -------
    int
        change-point location
    """
    if use_prob:
        ma = np.convolve(x[:, 1], np.ones(width) / width, mode="valid")
    else:
        ma = np.convolve(x, np.ones(width) / width, mode="valid")

    return np.argmax(ma) + int(np.round((width + n) / 2))


# %%
def DataSetGen(N_sub, n, mean_arg, var_arg, slope_arg, n_trim, seed=2022):
    """
    This function generates the simulation dataset for change in mean, in variance and change in non-zero slope. For more details, see Table S1 in supplement of "Automatic Change-point Detection in Time Series via Deep Learning" (Jie et al. ,2023)

    Parameters
    ----------
    N_sub : int
        the sample size of each class
    n : int
        the length of time series
    mean_arg : array
        the hyperparameters for generating data of change in mean and null
    var_arg : array
        the hyperparameters for generating data of change in variance and null
    slope_arg : array
        the hyperparameters for generating data of change in slope and null
    n_trim : int
        the trim size
    seed : int, optional
        the random seed, by default 2022

    Returns
    -------
    dictionary
        the simulation data and corresponding changes
    """
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
        if (
            abs_diff >= lower_bound_abs_mean_difference
            and abs_diff <= upper_bound_abs_mean_difference
        ):
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
        x_mean_1[i, :] = GenDataMean(1, n, cp=cp_mean[i], mu=mu_1, sigma=sigma_1)

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
        if (
            abs_diff >= lower_bound_abs_std_difference
            and abs_diff <= upper_bound_abs_std_difference
        ):
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
        x_var_1[i, :] = GenDataVariance(1, n, cp=cp_var[i], mu=mu_2, sigma=sigma_2)

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
        if (
            abs_diff >= lower_bound_abs_slope_difference
            and abs_diff <= upper_bound_abs_slope_difference
        ):
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
    data_x = np.concatenate((x_mean_0, x_mean_1, x_var_1, x_slope_0, x_slope_1), axis=0)
    return {
        "data_x": data_x,
        "cp_mean": cp_mean,
        "cp_var": cp_var,
        "cp_slope": cp_slope,
        "mu_para": mu_para,
        "sigma_para": sigma_para,
        "slopes_para": slopes_para,
    }


def seqPlot(sequences_list, cp_list, label_list, y_pos=0.93):
    """
    This function plots the sequence given change-points and label list.

    Parameters
    ----------
    sequences_list : DataFrame
        the time series
    cp_list : list
        the list of change-point
    label_list : list
        the list of labels
    y_pos : float, optional
        the position of y, used in matplotlib, by default 0.93
    """
    for seq, cp, label in zip(sequences_list, cp_list, label_list):
        seq.reset_index(drop=True, inplace=True)
        plt.figure()
        axes = seq.plot(y=["x", "y", "z"], figsize=(15, 6))
        axes.vlines(cp[0:-1], 0, 1, transform=axes.get_xaxis_transform(), colors="r")
        xlim = axes.get_xlim()
        cp = np.insert(cp, 0, xlim[0])
        x_range = np.diff(xlim)
        for i in range(len(label)):
            str_state = label["state"][i]
            if i == 0:
                x_pos = (np.mean(cp[i : i + 2]) - xlim[0]) / x_range
            else:
                x_pos = (np.mean(cp[i : i + 2]) - xlim[0] / 2) / x_range
            axes.text(x_pos, y_pos, str_state, transform=axes.transAxes)


def get_key(y_pred, label_dict):
    """
    To get the labels according to the predict value

    Parameters
    ----------
    y_pred : int
        the value of prediction
    label_dict : dict
        the lable dictionary

    Returns
    -------
    list
        the label list
    """
    label_str = list()
    for value in y_pred:
        key = [key for key, val in label_dict.items() if val == value]
        label_str.append(key[0])

    return label_str


def get_label_hasc(model, x_test, label_dict):
    """
    This function gets the predicted label for the HASC data

    Parameters
    ----------
    model : tensorflow model
        The trained tensorflow model
    x_test : 2D array
        The array of test dataset
    label_dict : dict
        The label dictionary

    Returns
    -------
    arrays
        two arrays, one is predicted label, the other is probabilities.
    """
    pred_prob = tf.math.softmax(model.predict(x_test))
    y_pred = np.argmax(pred_prob, axis=1)
    label_str = get_key(y_pred, label_dict)
    return label_str, pred_prob.numpy()
