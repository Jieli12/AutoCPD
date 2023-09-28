"""
Author         : Jie Li, Department of Statistics, London School of Economics.
Date           : 2023-09-25 11:11:18
Last Revision  : 2023-09-28 23:45:28
Last Author    : Jie Li
File Path      : /AutoCPD/test/test_load_pretrained_model.py
Description    :








Copyright (c) 2023 by Jie Li, j.li196@lse.ac.uk
All Rights Reserved.
"""
import os
import pathlib
import posixpath

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.signal import find_peaks

import autocpd
from autocpd.pre_trained_model import load_pretrained_model
from autocpd.utils import get_label_hasc, seqPlot

root_path = os.path.dirname(autocpd.__file__)
model_path = pathlib.Path(root_path, "Demo", "model")

model = load_pretrained_model(model_path)
model.summary()

datapath = pathlib.Path(root_path, "HASC")
np.random.seed(2022)  # numpy seed fixing
tf.random.set_seed(2022)  # tensorflow seed fixing
subjects = [
    "person101",
    "person102",
    "person103",
    "person104",
    "person105",
    "person106",
    "person107",
]
# %% load the trained model
label_dict = np.load(
    pathlib.Path(root_path, "Demo", "label_dict.npy"), allow_pickle=True
)
label_dict = label_dict.item()

# %%
subject = subjects[6]
subject_path = posixpath.join(datapath, subject)
# get the csv files
all_files = os.listdir(subject_path)
csv_files = list(filter(lambda f: f.endswith(".csv"), all_files))
csv_files = list(filter(lambda f: f.startswith("HASC"), csv_files))
sequences_list = list()
cp_list = list()
label_list = list()
## data preprocessing
for ind, fname in enumerate(csv_files):
    print(ind)
    print(fname)
    fname_label = fname.replace("-acc.csv", ".label")
    fname_label = posixpath.join(subject_path, fname_label)
    fname = posixpath.join(subject_path, fname)
    # load the labels
    label = pd.read_csv(
        fname_label, comment="#", delimiter=",", names=["start", "end", "state"]
    )
    label_list.append(label)
    num_consecutive_states = label.shape[0]
    # load the dataset
    data = pd.read_csv(fname, comment="#", delimiter=",", names=["time", "x", "y", "z"])
    cp = np.zeros((num_consecutive_states,))
    for ind in range(num_consecutive_states):
        s = label["start"][ind]
        e = label["end"][ind]
        state = label["state"][ind:]
        logical0 = (data["time"] >= s) & (data["time"] <= e)
        n1 = sum(logical0)
        cp[ind] = n1
        if ind == 0:
            data_trim = data[logical0]
        else:
            data_trim = pd.concat([data_trim, data[logical0]], axis=0)

    sequences_list.append(data_trim)
    cp_list.append(np.cumsum(cp))


# ## plot the raw datasets

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
    test_seq_0[i, :, :] = seq[step * i : step * i + length, :]


test2_seq_0 = np.square(test_seq_0)
TS_test = np.concatenate([test_seq_0, test2_seq_0], axis=2)
datamin = np.min(TS_test, axis=(1, 2), keepdims=True)
datamax = np.max(TS_test, axis=(1, 2), keepdims=True)
x_test = 2 * (TS_test - datamin) / (datamax - datamin) - 1
# %% predict
x_test = np.transpose(x_test, (0, 2, 1))
label_list, prob_list = get_label_hasc(model, x_test, label_dict)

# label-based
trans_sym = "->"
L = np.zeros((len(label_list),), dtype=np.int32)
for i, lab in enumerate(label_list):
    if trans_sym in lab:
        L[i] = 1

width = 700
L_bar = np.convolve(L, np.ones(width) / width, mode="valid")
peaks, _ = find_peaks(L_bar, height=0.5, distance=400)
cp_est = peaks + width

# %%
seq1 = sequences_list[index_csv]
y_pos = 0.93
cp = cp_list[index_csv]
seq1.reset_index(drop=True, inplace=True)
plt.figure()
axes = seq1.plot(y=["x", "y", "z"], figsize=(15, 6))
axes.vlines(cp[0:-1], 0, 1, transform=axes.get_xaxis_transform(), colors="r")
axes.vlines(cp_est, 0, 1, transform=axes.get_xaxis_transform(), colors="b")
xlim = axes.get_xlim()
cp = np.insert(cp, 0, xlim[0])
x_range = np.diff(xlim)
for i in range(len(label)):
    str_state = label["state"][i]
    if i == 0:
        x_pos = (np.mean(cp[i : i + 2]) - xlim[0]) / x_range
    else:
        x_pos = (np.mean(cp[i : i + 2]) - xlim[0] / 2) / x_range
    axes.text(x_pos, y_pos, str_state, transform=axes.transAxes, fontsize=16)

plt.xlabel("Time", fontsize=20)
plt.ylabel("Signal", fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=11, loc="upper right")
plt.savefig("./figs/HASCSubject7Seq1.png")
# %%
