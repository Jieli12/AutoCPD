# AutoCPD -- Automatic Change-Point Detection in Time Series via Deep Learning

[![PyPI version](https://badge.fury.io/py/autocpd.svg)](https://badge.fury.io/py/autocpd)  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) ![PyPI - Downloads](https://img.shields.io/pypi/dm/autocpd) ![Read the Docs](https://img.shields.io/readthedocs/autocpd)

`AutoCPD` is a Python package for detecting the multiple change-points and change-types in time series using deep neural networks.

**How to cite?** If you are interested in `AutoCPD` and use it in academic publications, please cite:

```bibtex
@article{JieAutoCPD2023,
	author  = {Li, Jie and Fearnhead, Paul and Fryzlewicz, Piotr and Wang, Tengyao},
	journal = {Journal of Royal Statistical Society, Series B (discussion, to appear)},
	title   = {Automatic Change-Point Detection in Time Series via Deep Learning},
	pages   = {arxiv:2211.03860},
	year    = {2023},
}
```

## Installation

Install via pip with

```bash
python3 -m pip install autocpd
```

**Note**: This package requires [tensorflow>=2.7](https://www.tensorflow.org/install) and [tensorflow-docs](https://github.com/tensorflow/docs), please install these two libraries in your virtual environment. For other requirements on the Python libraries, see [AutoCPD](https://autocpd.readthedocs.io/en/latest/index.html).

## Basic Usage

Please refer to the online [documentations](https://autocpd.readthedocs.io/en/latest/) for each module in `AutoCPD`. The two important functions to construct shallow and deep neural networks are `general_simple_nn()` and `general_deep_nn()` respectively. For `general_simple_nn()`, one can specify the number of layers, the width vector and the number of classes, see more details [here](https://autocpd.readthedocs.io/en/latest/autocpd.html#autocpd.neuralnetwork.simple_nn). For the function `general_deep_nn()`, one can specify the number of transformations, kernel size, filter size, number of classes, number of [residual blocks](https://autocpd.readthedocs.io/en/latest/autocpd.html#autocpd.neuralnetwork.resblock), etc, see more arguments description [here](https://autocpd.readthedocs.io/en/latest/autocpd.html#autocpd.neuralnetwork.general_deep_nn). To call these functions, just import them into Python script as:

```python
from autocpd.neuralnetwork import general_deep_nn, general_simple_nn
```

## Examples

In this section, we demonstrate 3 examples to show how to train simple and deep neural networks, compile and train them and how to detect the change-points. Each example has a corresponding Python script which can be found in the folder [./test](https://github.com/Jieli12/AutoCPD/tree/master/test).

### Simple Neural Network

The Python script `test_simple_nn.py` consists of three parts:

* Data Generation
* Model Construction
* Model Compilation and Fitting

#### Data Generation

We set the length of time series $n=100$, the training data size $N=400$. The sample size of Class 0 (without change-point) and Class 1 (with only 1 change-point) are equal. This example considers detecting change in mean for the piecewise constant signal + Gaussian noise model. Other parameter settings can be found in our [paper](https://arxiv.org/abs/2211.03860). Here is the Python script for generating training data set:

```python
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
N_all = 400  # the sample size
B = np.sqrt(8 * np.log(n / epsilon) / n)
mu_L = 0
tau_bound = 2
B_bound = np.array([0.5, 1.5])
rho = 0.7
# %% main double for loop
N = int(N_all / 2)
#  generate the dataset for alternative hypothesis
np.random.seed(2022)  # numpy seed
tf.random.set_seed(2022)  # tensorflow seed
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
```

**Note:** `AutoCPD` also offers the utility functions to generate the training data set for change in variance and slope with/without Gaussian noise. For non-Gaussian noise, only Cauchy noise and AR(1) noise are available in the current version, please see more functions in `autocpd.utils` [module](https://autocpd.readthedocs.io/en/latest/autocpd.html#module-autocpd.utils).

#### Model Construction

For simplicity, we set the number of hidden layers $l=2$, the width vector $m=50$ (when $m$ is a scalar, all hidden layers have the same nodes $m$).

```python
model_name = current_file + "n" + str(n) + "N" + str(2 * N) + "m" + str(m)
print(model_name)
# build the model
l = 2
m = 50
model = general_simple_nn(n=n, l=l, m=m, num_classes=2, model_name=model_name)
model.summary()
```

The architecture of this simple neural network is displayed below:

```python
test_simple_nnn100N400m50
Model: "test_simple_nnn100N400m50"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 Input (InputLayer)          [(None, 100)]             0
                                                                 
 dense (Dense)               (None, 50)                5050
                                                                 
 dense_1 (Dense)             (None, 50)                2550
                                                                 
 dense_2 (Dense)             (None, 2)                 102
                                                                 
=================================================================
Total params: 7,702
Trainable params: 7,702
Non-trainable params: 0
_________________________________________________________________
```

#### Model Compilation and Fitting

The function `compile_and_fit()` can compile and fit the model, furthermore, we also employ the [learning rate decay method](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/InverseTimeDecay) to schedule the learning rate according to the epochs in this function.

```python
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
```

**Note:**  The default value of `validation_split` in `compile_and_fit()` is 0.2, which means that in this example, the validation data set contains 80 samples.  Because we use the module `tensorflow-docs` to record the training history and plot the validation accuracy,  please check [tensorflow-docs](https://github.com/tensorflow/docs) is correctly installed and loaded before running Python script. It takes less than 3 minutes (depending on the CPU) to train the simple neural network on a laptop. The training and validation accuracies are displayed in the following figure:

![example1](https://github.com/Jieli12/AutoCPD/raw/master/test/figs/test_simple_nnn100N400m50%2Bacc.png)

### Deep Neural Network

In this section, we introduce how to construct a deep neural network, train it to classify multiple categories and test it on one unseen dataset. The code can be found in the Python script [./test/test_deep_nn.py](https://github.com/Jieli12/AutoCPD/tree/master/test). Compared with `test_simple_nn.py`, the script`test_test_nn.py` consists of four steps: data generation, model construction, model compilation and fitting and Model prediction. To be concise, we will omit the data generation, model compilation and fitting.

In this example, we try to construct a deep neural network with 3 residual blocks followed by 5 hidden layers, the width vector of hidden layers is $[50,40,30,20,10]$. The dataset can be grouped into 3 categories: change in variance only, no change in non-zero slope and change in slope labelled by 0, 1 and 2 respectively. The length of time series $n$ is 400, each class has 500 observations.

```python
np.random.seed(2022)  # numpy seed
tf.random.set_seed(2022)  # tensorflow seed
n = 400  # the length of time series
N_sub = 3000
num_dataset = 3
labels = [0, 1, 2]
num_classes = len(set(labels))

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
model_name = current_file + "n" + str(n) + "N" + str(N_sub) + "L" + str(3)
print(model_name)
# build the model
m = np.array([50, 40, 30, 20, 10])
num_resblock = 3
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
```

The architecture of the neural network with 3 residual blocks is displayed as below:

```python
test_deep_nnn400N3000L3
Model: "test_deep_nnn400N3000L3"
__________________________________________________________________________________________________
 Layer (type)                   Output Shape         Param #     Connected to
==================================================================================================
 Input (InputLayer)             [(None, 6, 400)]     0           []
                                                                                                  
 reshape (Reshape)              (None, 6, 400, 1)    0           ['Input[0][0]']
                                                                                                  
 conv2d (Conv2D)                (None, 6, 400, 16)   80          ['reshape[0][0]']
                                                                                                  
 batch_normalization (BatchNorm  (None, 6, 400, 16)  64          ['conv2d[0][0]']
 alization)
                                                                                                  
 re_lu (ReLU)                   (None, 6, 400, 16)   0           ['batch_normalization[0][0]']
                                                                                                  
 max_pooling2d (MaxPooling2D)   (None, 3, 200, 16)   0           ['re_lu[0][0]']
                                                                                                  
 conv2d_1 (Conv2D)              (None, 3, 200, 16)   7696        ['max_pooling2d[0][0]']
                                                                                                  
 batch_normalization_1 (BatchNo  (None, 3, 200, 16)  64          ['conv2d_1[0][0]']
 rmalization)
                                                                                                  
 re_lu_1 (ReLU)                 (None, 3, 200, 16)   0           ['batch_normalization_1[0][0]']
                                                                                                  
 conv2d_2 (Conv2D)              (None, 3, 200, 16)   7696        ['re_lu_1[0][0]']
                                                                                                  
 batch_normalization_2 (BatchNo  (None, 3, 200, 16)  64          ['conv2d_2[0][0]']
 rmalization)
                                                                                                  
 add (Add)                      (None, 3, 200, 16)   0           ['max_pooling2d[0][0]',
                                                                  'batch_normalization_2[0][0]']
                                                                                                  
 re_lu_2 (ReLU)                 (None, 3, 200, 16)   0           ['add[0][0]']
                                                                                                  
 conv2d_3 (Conv2D)              (None, 3, 200, 16)   7696        ['re_lu_2[0][0]']
                                                                                                  
 batch_normalization_3 (BatchNo  (None, 3, 200, 16)  64          ['conv2d_3[0][0]']
 rmalization)
                                                                                                  
 re_lu_3 (ReLU)                 (None, 3, 200, 16)   0           ['batch_normalization_3[0][0]']
                                                                                                  
 conv2d_4 (Conv2D)              (None, 3, 200, 16)   7696        ['re_lu_3[0][0]']
                                                                                                  
 batch_normalization_4 (BatchNo  (None, 3, 200, 16)  64          ['conv2d_4[0][0]']
 rmalization)
                                                                                                  
 add_1 (Add)                    (None, 3, 200, 16)   0           ['re_lu_2[0][0]',
                                                                  'batch_normalization_4[0][0]']
                                                                                                  
 re_lu_4 (ReLU)                 (None, 3, 200, 16)   0           ['add_1[0][0]']
                                                                                                  
 conv2d_5 (Conv2D)              (None, 3, 200, 16)   7696        ['re_lu_4[0][0]']
                                                                                                  
 batch_normalization_5 (BatchNo  (None, 3, 200, 16)  64          ['conv2d_5[0][0]']
 rmalization)
                                                                                                  
 re_lu_5 (ReLU)                 (None, 3, 200, 16)   0           ['batch_normalization_5[0][0]']
                                                                                                  
 conv2d_6 (Conv2D)              (None, 3, 200, 16)   7696        ['re_lu_5[0][0]']
                                                                                                  
 batch_normalization_6 (BatchNo  (None, 3, 200, 16)  64          ['conv2d_6[0][0]']
 rmalization)
                                                                                                  
 add_2 (Add)                    (None, 3, 200, 16)   0           ['re_lu_4[0][0]',
                                                                  'batch_normalization_6[0][0]']
                                                                                                  
 re_lu_6 (ReLU)                 (None, 3, 200, 16)   0           ['add_2[0][0]']
                                                                                                  
 global_average_pooling2d (Glob  (None, 16)          0           ['re_lu_6[0][0]']
 alAveragePooling2D)
                                                                                                  
 dense (Dense)                  (None, 50)           850         ['global_average_pooling2d[0][0]'
                                                                 ]
                                                                                                  
 dropout (Dropout)              (None, 50)           0           ['dense[0][0]']
                                                                                                  
 dense_1 (Dense)                (None, 40)           2040        ['dropout[0][0]']
                                                                                                  
 dropout_1 (Dropout)            (None, 40)           0           ['dense_1[0][0]']
                                                                                                  
 dense_2 (Dense)                (None, 30)           1230        ['dropout_1[0][0]']
                                                                                                  
 dropout_2 (Dropout)            (None, 30)           0           ['dense_2[0][0]']
                                                                                                  
 dense_3 (Dense)                (None, 20)           620         ['dropout_2[0][0]']
                                                                                                  
 dropout_3 (Dropout)            (None, 20)           0           ['dense_3[0][0]']
                                                                                                  
 dense_4 (Dense)                (None, 10)           210         ['dropout_3[0][0]']
                                                                                                  
 dense_5 (Dense)                (None, 3)            33          ['dense_4[0][0]']
                                                                                                  
==================================================================================================
Total params: 51,687
Trainable params: 51,463
Non-trainable params: 224
__________________________________________________________________________________________________
```

There are 51,463 trainable parameters in the above neural network. When the number of residual blocks increases, the number of trainable parameters increases dramatically. As a result, it will take more time to obtain a well-trained neural network. We recommend using GPU server to speed up training neural networks. For instance, it only took 7 minutes to train the above neural network on an HP laptop with Intel Core i7 and NVIDIA T500. The following figure shows the training and validation accuracies of neural networks. After 100 epochs, the validation accuracy is close to 93\%.

![example2](https://github.com/Jieli12/AutoCPD/raw/master/test/figs/test_deep_nnn400N3000L3%2Bacc.png)

The following figure shows The confusion matrix of prediction, the accuracy is 93.56\%.

![example3](https://github.com/Jieli12/AutoCPD/raw/master/test/figs/test_deep_nnn400N3000L3Confusion_matrix.png)

**Note:** To improve the accuracy of validation, one can either increase the number of residual blocks or increase the number of epochs.

### Load Pretrained Deep Neural Network

The script `test_load_pretrained_model.py` demonstrates how to load the pre-trained deep neural network with 21 residual blocks for [HASC](http://hasc.jp/hc2011/index-en.html) data analysis as described in [Jie et al. (2023)](https://arxiv.org/abs/2211.03860). The pre-trained model, named `Demo` in this example, was trained on [Lancaster HEC cluster](https://www.lancaster.ac.uk/iss/info/IThandouts/hec/HEC-flyer.pdf) which has NVIDIA V100 card. To load the pre-trained model, one can just add the following code at the top of Python script.

```python
import os
import pathlib
import posixpath

import autocpd
from autocpd.pre_trained_model import load_pretrained_model

root_path = os.path.dirname(autocpd.__file__)
model_path = pathlib.Path(root_path, "Demo", "model")

model = load_pretrained_model(model_path)
```

Alternatively, one can re-train the above deep neural network on your available GPU server by using the script [./test/RealDataHASC.py](https://github.com/Jieli12/AutoCPD/blob/master/test/RealDataHASC.py). Please note that, even on 1 NVIDIA V100 GPU card, it still needs 1~2 hours to train the deep neural network.

To predict the change-points, please run the following command (<=2 minutes):

```python
cd ./test
python test_load_pretrained_model.py
```

You can obtain the following figure:

![example4](https://github.com/Jieli12/AutoCPD/raw/master/test/figs/HASCSubject7Seq1.png)
