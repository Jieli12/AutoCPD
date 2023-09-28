"""
Author         : Jie Li, Department of Statistics, London School of Economics.
Date           : 2023-09-20 12:37:17
Last Revision  : 2023-09-28 18:52:17
Last Author    : Jie Li
File Path      : /AutoCPD/test/test_deep_nn.py
Description    :








Copyright (c) 2023 by Jie Li, j.li196@lse.ac.uk
All Rights Reserved.
"""
# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import tensorflow_docs.modeling as tfdoc_model
import tensorflow_docs.plots as tfdoc_plot
from sklearn.model_selection import train_test_split

from autocpd.neuralnetwork import compile_and_fit, general_deep_nn
from autocpd.utils import DataSetGen, Transform2D2TR

# %%
np.random.seed(2022)  # numpy seed fixing
tf.random.set_seed(2022)  # tensorflow seed fixing
n = 400  # the length of time series
N_sub = 3000
n_trim = 40
mean_arg = np.array([0.7, 5, -5, 1.2, 0.6])
var_arg = np.array([0, 0.7, 0.3, 0.4, 0.2])
slope_arg = np.array([0.5, 0.025, -0.025, 0.03, 0.015])
dataset = DataSetGen(N_sub, n, mean_arg, var_arg, slope_arg, n_trim)
data_x = dataset["data_x"]
# delete change in variance and no change in non-zero slope.
data_x = np.delete(data_x, np.arange(0 * N_sub, 2 * N_sub), 0)
# %% normalization
data_x = Transform2D2TR(data_x, rescale=True, times=3)
num_dataset = 3
labels = [0, 1, 2]
num_classes = len(set(labels))
data_y = np.repeat(labels, N_sub).reshape((N_sub * num_dataset, 1))

(
    x_train,
    x_test,
    y_train,
    y_test,
) = train_test_split(data_x, data_y, train_size=0.8, random_state=42)
#  setup the tensorboard
file_path = Path(__file__)
current_file = file_path.stem
print(current_file)
logdir = Path("tensorboard_logs", "Trial")

learning_rate = 1e-3
epochs = 100
batch_size = 64
dropout_rate = 0.3
n_filter = 16
n = x_train.shape[-1]
num_tran = x_train.shape[1]
kernel_size = (num_tran // 2, 10)
num_classes = 3

# %%
num_resblock = 3
model_name = current_file + "n" + str(n) + "N" + str(N_sub) + "L" + str(num_resblock)
print(model_name)
# build the model
m = np.array([50, 40, 30, 20, 10])
model = general_deep_nn(
    n,
    num_tran,
    kernel_size,
    n_filter,
    dropout_rate,
    num_classes,
    num_resblock,
    m,
    5,
    model_name=model_name,
)
model.summary()

size_histories = {}
epochdots = tfdoc_model.EpochDots()
size_histories[model_name] = compile_and_fit(
    model,
    x_train,
    y_train,
    batch_size,
    learning_rate,
    model_name,
    logdir,
    epochdots,
    validation_split=0.25,
    max_epochs=epochs,
)
plotter = tfdoc_plot.HistoryPlotter(metric="accuracy", smoothing_std=10)
plt.figure(figsize=(10, 8))
plotter.plot(size_histories)
acc_name = model_name + "+acc.png"
acc_path = Path(logdir, model_name, acc_name)
plt.savefig(acc_path)


model_path = Path(logdir, model_name, "model")
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
    fmt="g",
)
plt.xlabel("Prediction")
plt.ylabel("Label")
cm_name = model_name + "Confusion_matrix.png"
cm_path = Path(logdir, model_name, cm_name)
plt.savefig(cm_path)

# save the confusion matrix
path_confusion_matrix = Path(logdir, model_name, "confusion_matrix")
np.save(path_confusion_matrix, confusion_mtx)

# %%
