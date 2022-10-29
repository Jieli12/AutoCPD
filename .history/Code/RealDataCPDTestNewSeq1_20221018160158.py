"""
Author         : Jie Li, Department of Statistics, London School of Economics.
Date           : 2022-04-25 21:45:38
Last Revision  : 2022-05-30 10:21:30
Last Author    : Jie Li
File Path      : /AI-assisstedChangePointDetection/Python/RealDataCPDTestNew.py
Description    :








Copyright (c) 2022 by Jie Li, j.li196@lse.ac.uk
All Rights Reserved.
"""
# %%
import pathlib
from collections import Counter
# %%
from itertools import groupby
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import tensorflow_docs.plots
from keras import layers, losses, metrics, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sympy import diff

from utilsMultimode import *

# %%
np.random.seed(2022)  # numpy seed fixing
tf.random.set_seed(2022)  # tensorflow seed fixing
datapath = "../datasets/RealData/0_sequence/"
subjects = [
	"person101",
	"person102",
	"person103",
	"person104",
	"person105",
	"person106",
	"person107"
]
# %% load the trained model
logdir = Path("tensorboard_logs", "Trial")
model_name = 'RealDataKS25'
model_path = Path(logdir, model_name, 'model')
model = tf.keras.models.load_model(model_path)
model.summary()
label_dict = np.load(
	Path(logdir, model_name, 'label_dict.npy'), allow_pickle=True
)
label_dict = label_dict.item()
# %%  load the sequences from subject 7, data preprocessing and plot with CPs

# load the sequences, there are 3 sequences.
subject = subjects[6]
subject_path = datapath + subject
# get the csv files
all_files = os.listdir(subject_path)
csv_files = list(filter(lambda f: f.endswith('.csv'), all_files))
csv_files = list(filter(lambda f: f.startswith('HASC'), csv_files))
sequences_list = list()
cp_list = list()
label_list = list()
## data preprocessing
for ind, fname in enumerate(csv_files):
	print(ind)
	print(fname)
	fname_label = fname.replace('-acc.csv', '.label')
	fname_label = posixpath.join(subject_path, fname_label)
	fname = posixpath.join(subject_path, fname)
	# load the labels
	label = pd.read_csv(
		fname_label,
		comment="#",
		delimiter=",",
		names=['start', 'end', 'state']
	)
	label_list.append(label)
	num_consecutive_states = label.shape[0]
	# load the dataset
	data = pd.read_csv(
		fname, comment="#", delimiter=",", names=['time', 'x', 'y', 'z']
	)
	cp = np.zeros((num_consecutive_states,))
	for ind in range(num_consecutive_states):
		s = label['start'][ind]
		e = label['end'][ind]
		state = label['state'][ind:]
		logical0 = (data['time'] >= s) & (data['time'] <= e)
		n1 = sum(logical0)
		cp[ind] = n1
		if ind == 0:
			data_trim = data[logical0]
		else:
			data_trim = pd.concat([data_trim, data[logical0]], axis=0)

	sequences_list.append(data_trim)
	cp_list.append(np.cumsum(cp))


## plot the raw datasets
def seqPlot(sequences_list, cp_list, label_list, y_pos=0.93):
	for seq, cp, label in zip(sequences_list, cp_list, label_list):
		seq.reset_index(drop=True, inplace=True)
		plt.figure()
		axes = seq.plot(y=['x', 'y', 'z'], figsize=(15, 6))
		axes.vlines(
			cp[0:-1], 0, 1, transform=axes.get_xaxis_transform(), colors='r'
		)
		xlim = axes.get_xlim()
		cp = np.insert(cp, 0, xlim[0])
		x_range = np.diff(xlim)
		for i in range(len(label)):
			str = label['state'][i]
			if i == 0:
				x_pos = (np.mean(cp[i:i + 2]) - xlim[0]) / x_range
			else:
				x_pos = (np.mean(cp[i:i + 2]) - xlim[0] / 2) / x_range
			axes.text(x_pos, y_pos, str, transform=axes.transAxes)


seqPlot(sequences_list, cp_list, label_list, y_pos=0.93)

# %%
index_csv = 0
seq = sequences_list[index_csv].to_numpy()[:, 1:]
length = 700
n_max = seq.shape[0]
step = 1
n = (n_max - length) // step + 1
test_seq_0 = np.zeros((n, length, 3))
for i in range(n):
	test_seq_0[i, :, :] = seq[step * i:step * i + length, :]


def get_key(y_pred, label_dict):
	label_str = list()
	for value in y_pred:
		key = [key for key, val in label_dict.items() if val == value]
		label_str.append(key[0])

	return label_str


def get_label(model, x_test, label_dict):
	y_pred = np.argmax(model.predict(x_test), axis=1)
	label_str = get_key(y_pred, label_dict)
	return label_str


test2_seq_0 = np.square(test_seq_0)
TS_test = np.concatenate([test_seq_0, test2_seq_0], axis=2)
datamin = TS_test.min(axis=(1, 2), keepdims=True)
datamax = TS_test.max(axis=(1, 2), keepdims=True)
x_test = 2 * (TS_test - datamin) / (datamax - datamin) - 1
# %% predict
x_test = np.transpose(x_test, (0, 2, 1))
label_list = get_label(model, x_test, label_dict)

trans_sym = '->'
num_each_group = []
state = []
for key, group in groupby(label_list):
	print(key)
	g = list(group)
	state.append(g[0])
	num_each_group.append(len(g))
df = pd.DataFrame(state, num_each_group)
Ind = np.cumsum(num_each_group)

state_filter = []
ind_filter = []
len_filter = []
threshold = 100
for s, ind, len0 in zip(state, Ind, num_each_group):
	print([s, ind])
	if trans_sym not in s and len0 > threshold:
		state_filter.append(s)
		ind_filter.append(ind)
		len_filter.append(len0)
int_est = np.zeros((12 - 1, 2))
for i in range(12 - 1):
	a = ind_filter[i] * step + length + step // 2
	b = (ind_filter[i + 1] - len_filter[i + 1]) * step - step // 2
	if a <= b:
		int_est[i, :] = [a, b]
	else:
		int_est[i, :] = [b, a]

# %%
print(int_est)
print(np.mean(int_est, axis=1))
np.cumsum(cp)
cp_est = np.round(np.mean(int_est, axis=1), decimals=0)
# %%
seq1 = sequences_list[index_csv]
y_pos = 0.93
cp = cp_list[index_csv]
seq1.reset_index(drop=True, inplace=True)
plt.figure()
axes = seq1.plot(y=['x', 'y', 'z'], figsize=(15, 6))
axes.vlines(cp[0:-1], 0, 1, transform=axes.get_xaxis_transform(), colors='r')
axes.vlines(cp_est, 0, 1, transform=axes.get_xaxis_transform(), colors='b')
xlim = axes.get_xlim()
cp = np.insert(cp, 0, xlim[0])
x_range = np.diff(xlim)
for i in range(len(label)):
	str = label['state'][i]
	if i == 0:
		x_pos = (np.mean(cp[i:i + 2]) - xlim[0]) / x_range
	else:
		x_pos = (np.mean(cp[i:i + 2]) - xlim[0] / 2) / x_range
	axes.text(x_pos, y_pos, str, transform=axes.transAxes)
# %%
mad = np.mean(np.abs(cp_est - cp[1:-1]))
print(mad)
n_min = min(np.diff(cp[1:]))
print(n_min)
mad / n_min
