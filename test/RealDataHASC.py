"""
Author         : Jie Li, Department of Statistics, London School of Economics.
Date           : 2023-09-28 22:29:18
Last Revision  : 2023-09-28 23:12:48
Last Author    : Jie Li
File Path      : /AutoCPD/test/RealDataHASC.py
Description    :








Copyright (c) 2023 by Jie Li, j.li196@lse.ac.uk
All Rights Reserved.
"""

# %%
import os
import pathlib
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_docs as tfdocs
from keras import losses, metrics
from sklearn.preprocessing import LabelEncoder

import autocpd
from autocpd.neuralnetwork import deep_nn
from autocpd.utils import ExtractSubject, labelSubject

# %%
# load the real dataset
root_path = os.path.dirname(autocpd.__file__)
datapath = pathlib.Path(root_path, "HASC")
# set the random seed
np.random.seed(2022)  # numpy seed
tf.random.set_seed(2022)  # tensorflow seed
subjects = [
    "person101",
    "person102",
    "person103",
    "person104",
    "person105",
    "person106",
    "person107",
]
length = 700
size = 15
size0 = 15

# %%
# extract the data
train_subjects = subjects[:6]

# extract the change-point from the training dataset
for i, subject in enumerate(train_subjects):
    print(subject)
    subject_path = pathlib.Path(datapath, subject)
    result = labelSubject(subject_path, length, size, num_trim=100)
    if i == 0:
        ts = result["ts"]
        label = result["label"]
        cp = result["cp"]
        id1 = [subject] * cp.shape[0]
    else:
        ts = np.concatenate([ts, result["ts"]], axis=0)
        cp = np.concatenate([cp, result["cp"]])
        label += result["label"]
        id1 += [subject] * result["cp"].shape[0]

# extract the no change-point from the training dataset
for i, subject in enumerate(train_subjects):
    print(subject)
    subject_path = pathlib.Path(datapath, subject)
    result0 = ExtractSubject(subject_path, length, size0)
    if i == 0:
        ts0 = result0["ts"]
        label0 = result0["label"]
        id0 = [subject] * ts0.shape[0]
    else:
        ts0 = np.concatenate([ts0, result0["ts"]], axis=0)
        label0 += result0["label"]
        id0 += [subject] * result0["ts"].shape[0]

ts_train = np.concatenate([ts, ts0], axis=0).copy()
label_train = label + label0
cp_train = cp.copy()
id_train = id1 + id0


TS_train = ts_train.copy()
CP_train = cp_train.copy()
LABEL_train = label_train.copy()

# check the number and frequency of labels
counts = Counter(LABEL_train)
print(counts)
len(counts)
# %% add square transformation
tstrain2 = np.square(TS_train)
TS_train = np.concatenate([TS_train, tstrain2], axis=2)
# rescale
datamin = np.min(TS_train, axis=(1, 2), keepdims=True)
datamax = np.max(TS_train, axis=(1, 2), keepdims=True)
TS_train = 2 * (TS_train - datamin) / (datamax - datamin) - 1

# %%
le = LabelEncoder()
label_train = le.fit_transform(LABEL_train)
# label_train = label_all[: len(LABEL_train)]
# shuffle the datasets
ind_train = np.random.permutation(TS_train.shape[0])

x_train = TS_train[ind_train, :, :]
y_train = label_train[ind_train]

# %% get the working path
current_file = pathlib.Path(__file__).stem
model_name = current_file
logdir = pathlib.Path("tensorboard_logs", "Trial")
# parameter settings
x_train = np.transpose(x_train, (0, 2, 1))
learning_rate = 1e-3
epochs = 400
batch_size = 32
dropout_rate = 0.3
n_filter = 16
n = x_train.shape[-1]
kernel_size = (3, 25)
num_classes = len(counts)
num_tran = TS_train.shape[-1]

# %%
model_name = "RealDataHACS"
m = np.array([50, 40, 30])
l = len(m)
model = deep_nn(
    n=n,
    n_trans=num_tran,
    kernel_size=kernel_size,
    n_filter=n_filter,
    dropout_rate=dropout_rate,
    n_classes=num_classes,
    m=m,
    l=l,
    model_name=model_name,
)
model.summary()
# %%
# reduce the learning rate over time
lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
    learning_rate, decay_steps=5000, decay_rate=1, staircase=False
)


def get_optimizer():
    return tf.keras.optimizers.Adam(lr_schedule)


def get_callbacks(name):
    name1 = pathlib.Path(name, "/log.csv")
    return [
        tfdocs.modeling.EpochDots(),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_sparse_categorical_crossentropy", patience=300, min_delta=1e-3
        ),
        tf.keras.callbacks.TensorBoard(pathlib.Path(logdir, name)),
        tf.keras.callbacks.CSVLogger(pathlib.Path(logdir, name1)),
    ]


def compile_and_fit(model, name, optimizer=None, max_epochs=10000):
    if optimizer is None:
        optimizer = get_optimizer()
    model.compile(
        optimizer=optimizer,
        loss=losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            metrics.SparseCategoricalCrossentropy(
                from_logits=True, name="sparse_categorical_crossentropy"
            ),
            "accuracy",
        ],
    )
    history = model.fit(
        x_train,
        y_train,
        epochs=max_epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=get_callbacks(name),
    )
    return history


size_histories = {}
size_histories[model_name] = compile_and_fit(
    model,
    model_name,
    max_epochs=epochs,
)
plt.figure(figsize=(10, 8))
plotter = tfdocs.plots.HistoryPlotter(metric="accuracy", smoothing_std=10)
plotter.plot(size_histories)
acc_name = model_name + "+acc.png"
acc_path = pathlib.Path(logdir, model_name, acc_name)
plt.savefig(acc_path)
model_path = pathlib.Path(logdir, model_name, "model")
model.save(model_path)

# %%
label_num = np.arange(num_classes)
label_string = le.inverse_transform(label_num)
label_dict = dict(zip(label_string, label_num))
cm_name = "label_dict"
cm_path = pathlib.Path(logdir, model_name, cm_name)
np.save(cm_path, label_dict)
