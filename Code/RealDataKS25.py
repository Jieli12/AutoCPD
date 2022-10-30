"""
Author         : Jie Li, Department of Statistics, London School of Economics.
Date           : 2022-10-18 16:01:59
Last Revision  : 2022-10-30 08:28:05
Last Author    : Jie Li
File Path      : /AutoCPD/Code/RealDataKS25.py
Description    :








Copyright (c) 2022 by Jie Li, j.li196@lse.ac.uk
All Rights Reserved.
"""
# %%
import pathlib
from collections import Counter
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

from utils import *

# %%
# load the real dataset

# set the random seed
np.random.seed(2022)  # numpy seed fixing
tf.random.set_seed(2022)  # tensorflow seed fixing
datapath = "../datasets/HASC/"
subjects = [
	"person101",
	"person102",
	"person103",
	"person104",
	"person105",
	"person106",
	"person107"
]
length = 700
size = 15
size0 = 15

# %%
# extract the data

train_subjects = subjects[:6]
test_subjects = subjects[6:]

# extract the change-point from the training dataset
for i, subject in enumerate(train_subjects):
	print(subject)
	subject_path = datapath + subject
	result = labelSubject(subject_path, length, size, num_trim=100)
	if i == 0:
		ts = result['ts']
		label = result['label']
		cp = result['cp']
		id = [subject] * cp.shape[0]
	else:
		ts = np.concatenate([ts, result['ts']], axis=0)
		cp = np.concatenate([cp, result['cp']])
		label += result['label']
		id += [subject] * result['cp'].shape[0]

# extract the no change-point from the training dataset
for i, subject in enumerate(train_subjects):
	print(subject)
	subject_path = datapath + subject
	result0 = ExtractSubject(subject_path, length, size0)
	if i == 0:
		ts0 = result0['ts']
		label0 = result0['label']
		id0 = [subject] * ts0.shape[0]
	else:
		ts0 = np.concatenate([ts0, result0['ts']], axis=0)
		label0 += result0['label']
		id0 += [subject] * result0['ts'].shape[0]

ts_train = np.concatenate([ts, ts0], axis=0).copy()
label_train = label + label0
cp_train = cp.copy()
id_train = id + id0

# extract the change-point from the test dataset
for i, subject in enumerate(test_subjects):
	print(subject)
	subject_path = datapath + subject
	result = labelSubject(subject_path, length, size, num_trim=100)
	if i == 0:
		ts = result['ts']
		label = result['label']
		cp = result['cp']
		id = [subject] * cp.shape[0]
	else:
		ts = np.concatenate([ts, result['ts']], axis=0)
		cp = np.concatenate([cp, result['cp']])
		label += result['label']
		id += [subject] * result['cp'].shape[0]

# extract the no change-point from the test dataset
for i, subject in enumerate(test_subjects):
	print(subject)
	subject_path = datapath + subject
	result0 = ExtractSubject(subject_path, length, size0)
	if i == 0:
		ts0 = result0['ts']
		label0 = result0['label']
		id0 = [subject] * ts0.shape[0]
	else:
		ts0 = np.concatenate([ts0, result0['ts']], axis=0)
		label0 += result0['label']
		id0 += [subject] * result0['ts'].shape[0]

ts_test = np.concatenate([ts, ts0], axis=0).copy()
label_test = label + label0
cp_test = cp.copy()
id_test = id + id0

TS_train = ts_train.copy()
CP_train = cp_train.copy()
LABEL_train = label_train.copy()
TS_test = ts_test.copy()
LABEL_test = label_test.copy()

# check the number and frequency of labels
counts = Counter(LABEL_train)
print(counts)
len(counts)
# %% add square transformation
tstrain2 = np.square(TS_train)
tstest2 = np.square(TS_test)
TS_train = np.concatenate([TS_train, tstrain2], axis=2)
TS_test = np.concatenate([TS_test, tstest2], axis=2)
# %%
# standardize, Temporarily no transformation as the real data includes x, y and z accelarations

# rescale
datamin = TS_train.min(axis=(1, 2), keepdims=True)
datamax = TS_train.max(axis=(1, 2), keepdims=True)
TS_train = 2 * (TS_train - datamin) / (datamax - datamin) - 1

datamin = TS_test.min(axis=(1, 2), keepdims=True)
datamax = TS_test.max(axis=(1, 2), keepdims=True)
TS_test = 2 * (TS_test - datamin) / (datamax - datamin) - 1

# %%
# plot one case
colors = [c for c, _ in mcolors.TABLEAU_COLORS.items()]
fig, axs = plt.subplots(3, 1, figsize=(15, 10))
for i, ax in enumerate(axs.flat):
	ax.plot(TS_train[0, :, i], colors[i])
plt.show()
le = LabelEncoder()
label_all = le.fit_transform(np.concatenate([LABEL_train, LABEL_test]))
label_train = label_all[:len(LABEL_train)]
label_test = label_all[len(LABEL_train):]
# shuffle the datasets
ind_train = np.random.permutation(TS_train.shape[0])
ind_test = np.random.permutation(TS_test.shape[0])

x_train = TS_train[ind_train, :, :]
x_test = TS_test[ind_test, :, :]
y_train = label_train[ind_train]
y_test = label_test[ind_test]

# %% get the working path
current_file = Path(__file__).stem
print(current_file)
model_name = current_file
logdir = Path("tensorboard_logs", "Trial")
# %% parameter settings
x_test = np.transpose(x_test, (0, 2, 1))
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

input = layers.Input(shape=(num_tran, n), name="Input")
x = layers.Reshape((num_tran, n, 1))(input)
x = layers.Conv2D(n_filter, 2, padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.MaxPooling2D((2, 2))(x)

x = resblock(x, kernel_size, filters=n_filter)
x = resblock(x, kernel_size, filters=n_filter)
x = resblock(x, kernel_size, filters=n_filter)

# n_filter = 2 * n_filter
x = resblock(x, kernel_size, filters=n_filter, strides=(1, 2))
x = resblock(x, kernel_size, filters=n_filter)
x = resblock(x, kernel_size, filters=n_filter)
x = resblock(x, kernel_size, filters=n_filter)

# n_filter = 2 * n_filter
x = resblock(x, kernel_size, filters=n_filter, strides=(1, 2))
x = resblock(x, kernel_size, filters=n_filter)
x = resblock(x, kernel_size, filters=n_filter)
x = resblock(x, kernel_size, filters=n_filter)
x = resblock(x, kernel_size, filters=n_filter)
x = resblock(x, kernel_size, filters=n_filter)

x = resblock(x, kernel_size, filters=n_filter, strides=(1, 2))
x = resblock(x, kernel_size, filters=n_filter)
x = resblock(x, kernel_size, filters=n_filter)
x = resblock(x, kernel_size, filters=n_filter)

x = resblock(x, kernel_size, filters=n_filter, strides=(1, 2))
x = resblock(x, kernel_size, filters=n_filter)
x = resblock(x, kernel_size, filters=n_filter)
x = resblock(x, kernel_size, filters=n_filter)

# x = layers.Flatten()(x)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(50, activation="relu", kernel_regularizer='l2')(x)
x = layers.Dropout(dropout_rate)(x)
x = layers.Dense(40, activation="relu", kernel_regularizer='l2')(x)
x = layers.Dropout(dropout_rate)(x)
x = layers.Dense(30, activation="relu", kernel_regularizer='l2')(x)
output = layers.Dense(num_classes)(x)
model = models.Model(input, output, name="ResNet30RealData")
model.summary()
# %%
# reduce the learning rate over time
lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
	learning_rate, decay_steps=5000, decay_rate=1, staircase=False
)


def get_optimizer():
	return tf.keras.optimizers.Adam(lr_schedule)


def get_callbacks(name):
	name1 = name + '/log.csv'
	return [
		tfdocs.modeling.EpochDots(),
		tf.keras.callbacks.EarlyStopping(
			monitor='val_sparse_categorical_crossentropy',
			patience=300,
			min_delta=1e-3
		),
		tf.keras.callbacks.TensorBoard(Path(logdir, name)),
		tf.keras.callbacks.CSVLogger(Path(logdir, name1)),
	]


def compile_and_fit(model, name, optimizer=None, max_epochs=10000):
	if optimizer is None:
		optimizer = get_optimizer()
	model.compile(
		optimizer=optimizer,
		loss=losses.SparseCategoricalCrossentropy(from_logits=True),
		metrics=[
			metrics.SparseCategoricalCrossentropy(
				from_logits=True, name='sparse_categorical_crossentropy'
			),
			"accuracy"
		]
	)
	model.summary()
	history = model.fit(
		x_train,
		y_train,
		epochs=max_epochs,
		batch_size=batch_size,
		validation_split=0.2,
		callbacks=get_callbacks(name)
	)
	return history


size_histories = {}
size_histories[model_name] = compile_and_fit(
	model,
	model_name,
	# optimizer=optimizers.Adam(learning_rate=learning_rate),
	max_epochs=epochs
)
plt.figure(figsize=(10, 8))
plotter = tfdocs.plots.HistoryPlotter(metric='accuracy', smoothing_std=10)
plotter.plot(size_histories)
acc_name = model_name + '+acc.png'
acc_path = Path(logdir, model_name, acc_name)
plt.savefig(acc_path)

model_path = Path(logdir, model_name, 'model')
model.save(model_path)

#  Confusion Matrix

model_pred = model.evaluate(x_test, y_test, verbose=2)
y_prob = np.max(model.predict(x_test), axis=1)
y_pred = np.argmax(model.predict(x_test), axis=1)
confusion_mtx = tf.math.confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
label_vec = range(0, num_classes)
sns.heatmap(
	confusion_mtx,
	cmap="YlGnBu",
	xticklabels=label_vec,
	yticklabels=label_vec,
	annot=True,
	fmt='g'
)
plt.xlabel('Prediction')
plt.ylabel('Label')
cm_name = model_name + '+Confusion_matrix.png'
cm_path = Path(logdir, model_name, cm_name)
plt.savefig(cm_path)
# %%
label_num = np.arange(num_classes)
label_string = le.inverse_transform(label_num)
label_dict = dict(zip(label_string, label_num))
cm_name = 'label_dict'
cm_path = Path(logdir, model_name, cm_name)
np.save(cm_path, label_dict)
# %%
