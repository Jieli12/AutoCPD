"""
Author         : Jie Li, Department of Statistics, London School of Economics.
Date           : 2022-10-29 16:02:04
Last Revision  : 2023-09-04 10:39:34
Last Author    : Jie Li
File Path      : /AutoCPD/Code/ResNetN1kE8tanhDecay10kScale.py
Description    :








Copyright (c) 2022 by Jie Li, j.li196@lse.ac.uk
All Rights Reserved.
"""
# %%

import pathlib
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from DataSetGen import *
from keras import layers, losses, metrics, models
# %%
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import train_test_split
from utils import *

# %%
# set the random seed
np.random.seed(2022)  # numpy seed fixing
tf.random.set_seed(2022)  # tensorflow seed fixing
N_sub = 2500
n = 400
n_trim = 40
mean_arg = np.array([0.7, 5, -5, 0.5, 0.25])
var_arg = np.array([0, 0.7, 0.3, 0.24, 0.12])
slope_arg = np.array([0.5, 0.025, -0.025, 0.012, 0.006])

dataset = DataSetGen0(N_sub, n, mean_arg, var_arg, slope_arg, n_trim)

data_x = dataset["data_x"]
mu_para = dataset["mu_para"]
sigma_para = dataset["sigma_para"]
slopes_para = dataset["slopes_para"]
cp_var = dataset["cp_var"]
cp_slope = dataset["cp_slope"]
cp_mean = dataset["cp_mean"]
# %% normalization
data_x = Transform2D2TR(data_x, rescale=True, times=3)
num_dataset = 5
labels = [0, 1, 2, 3, 4]
num_classes = len(set(labels))
data_y = np.repeat(labels, N_sub).reshape((N_sub * num_dataset, 1))
cp_non = np.zeros((N_sub,))
cp_all = np.concatenate((cp_non, cp_mean, cp_var, cp_non, cp_slope))
range = np.arange(N_sub * num_dataset)
x_train, x_test, y_train, y_test, cp_train, cp_test, ind_train, ind_test= train_test_split(
	data_x, data_y, cp_all, range, train_size=0.8, random_state=42
)
# %%
current_file = Path(__file__).stem
print(current_file)
model_name = current_file
logdir = Path("tensorboard_logs", "Trial")
# %%
learning_rate = 1e-3
epochs = 500
batch_size = 64
dropout_rate = 0.3
n_filter = 16
n = x_train.shape[-1]
num_tran = x_train.shape[1]
kernel_size = (num_tran // 2, 30)
num_classes = 5


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


# Note: the following network will cost several hours to train the residual neural network in GPU server.
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

x = resblock(x, kernel_size, filters=n_filter, strides=(1, 2))
x = resblock(x, kernel_size, filters=n_filter)
x = resblock(x, kernel_size, filters=n_filter)
x = resblock(x, kernel_size, filters=n_filter)
x = resblock(x, kernel_size, filters=n_filter)
x = resblock(x, kernel_size, filters=n_filter)

# n_filter = 2 * n_filter
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
x = layers.Dropout(dropout_rate)(x)
x = layers.Dense(20, activation="relu", kernel_regularizer='l2')(x)
x = layers.Dropout(dropout_rate)(x)
x = layers.Dense(10, activation="relu", kernel_regularizer='l2')(x)
output = layers.Dense(num_classes)(x)
model = models.Model(input, output, name=current_file)
model.summary()

# %%
lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
	learning_rate, decay_steps=10000, decay_rate=1, staircase=False
)


def get_optimizer():
	return tf.keras.optimizers.Adam(lr_schedule)


def get_callbacks(name):
	name1 = name + '/log.csv'
	return [
		tf.keras.callbacks.EarlyStopping(
			monitor='val_sparse_categorical_crossentropy',
			patience=800,
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
			"accuracy",
			metrics.SparseCategoricalCrossentropy(
				from_logits=True, name='sparse_categorical_crossentropy'
			)
		]
	)
	model.summary()
	history = model.fit(
		x_train,
		y_train,
		epochs=max_epochs,
		batch_size=batch_size,
		validation_split=0.25,
		callbacks=get_callbacks(name),
		verbose=2
	)
	return history


size_histories = {}
size_histories[model_name] = compile_and_fit(
	model,
	model_name,
	# optimizer=optimizers.Adam(learning_rate=learning_rate),
	max_epochs=epochs
)

# summarize history for accuracy
plt.figure(figsize=(10, 8))
plt.plot(size_histories[model_name].history['accuracy'])
plt.plot(size_histories[model_name].history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'])
acc_name = model_name + '+acc.png'
acc_path = Path(logdir, model_name, acc_name)
plt.savefig(acc_path)
plt.clf()

# summarize history for loss
plt.figure(figsize=(10, 8))
plt.plot(size_histories[model_name].history['loss'])
plt.plot(size_histories[model_name].history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'])
loss_name = model_name + '+loss.png'
loss_path = Path(logdir, model_name, loss_name)
plt.savefig(loss_path)
plt.clf()

model_path = Path(logdir, model_name, 'model')
model.save(model_path)

#  Confusion Matrix

model_pred = model.evaluate(x_test, y_test, verbose=2)
y_prob = np.max(model.predict(x_test), axis=1)
y_pred = np.argmax(model.predict(x_test), axis=1)
confusion_mtx = tf.math.confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
label_vec = np.arange(num_classes)
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
cm_name = model_name + 'Confusion_matrix.png'
cm_path = Path(logdir, model_name, cm_name)
plt.savefig(cm_path)

# save the confusion matrix
path_confusion_matrix = Path(logdir, model_name, 'confusion_matrix')
np.save(path_confusion_matrix, confusion_mtx)
