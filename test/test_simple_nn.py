"""
Author         : Jie Li, Department of Statistics, London School of Economics.
Date           : 2023-09-14 18:57:05
Last Revision  : 2023-09-28 11:37:24
Last Author    : Jie Li
File Path      : /AutoCPD/test/test_simple_nn.py
Description    :








Copyright (c) 2023 by Jie Li, j.li196@lse.ac.uk
All Rights Reserved.
"""
# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_docs.modeling as tfdoc_model
import tensorflow_docs.plots as tfdoc_plot
from sklearn.utils import shuffle

from autocpd.neuralnetwork import compile_and_fit, general_simple_nn
from autocpd.utils import DataGenAlternative, GenDataMean

# %%
n = 100  # the length of time series
epsilon = 0.05
m = 50
N_all = 400  # the sample size
B = np.sqrt(8 * np.log(n / epsilon) / n)
mu_L = 0
tau_bound = 2
B_bound = np.array([0.5, 1.5])
rho = 0.7
# parameters for neural network
learning_rate = 1e-3
epochs = 200
batch_size = 32
num_classes = 2
#  setup the tensorboard
file_path = Path(__file__)
current_file = file_path.stem
print(current_file)
logdir = Path("tensorboard_logs", "Trial")

# %% main double for loop
N = int(N_all / 2)
#  generate the dataset for alternative hypothesis
np.random.seed(2022)  # numpy seed fixing
tf.random.set_seed(2022)  # tensorflow seed fixing
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
tau_alt = result["tau_alt"]
mu_R_alt = result["mu_R_alt"]
#  generate dataset for null hypothesis
data_null = GenDataMean(N, n, cp=None, mu=(mu_L, mu_L), sigma=1)
data_all = np.concatenate((data_alt, data_null), axis=0)
y_all = np.repeat((1, 0), N).reshape((2 * N, 1))
tau_all = np.concatenate((tau_alt, np.repeat(0, N)), axis=0)
mu_R_all = np.concatenate((mu_R_alt, np.repeat(mu_L, N)), axis=0)
#  generate the training dataset and test dataset
x_train, y_train, tau_train, mu_R_train = shuffle(
    data_all, y_all, tau_all, mu_R_all, random_state=42
)
# %%
model_name = current_file + "n" + str(n) + "N" + str(2 * N) + "m" + str(m)
print(model_name)
# build the model
l = 2
model = general_simple_nn(n=n, l=l, m=m, num_classes=2, model_name=model_name)
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
