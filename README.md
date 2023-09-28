# AutoCPD -- Automatic Change-Point Detection in Time Series via Deep Learning

`AutoCPD` is a Python package for detecting the multiple change-points and change-types using deep neural networks.

## Installation

Install via pip with

```bash
python3 -m pip install autocpd
```

**Note**: This package requires [tensorflow>=2.7](https://www.tensorflow.org/install) and [tensorflow-docs](https://github.com/tensorflow/docs), please install these two libraries in your virtual environment. For other requirements on the Python libraries, see [AutoCPD]().

## Basic Usage

Please refer to the online [documentations](https://autocpd.readthedocs.io/en/latest/) for each module in `AutoCPD`. The two important functions to construct shallow and deep neural network are `general_simple_nn()` and `general_deep_nn()` respectively. For `general_simple_nn()`, one can specify the number of layers, the width vector and the number of classes, see more details [here](https://autocpd.readthedocs.io/en/latest/autocpd.html#autocpd.neuralnetwork.simple_nn). For the function `general_deep_nn()`, one can specify the number of transformations, kernel size, filter size, number of classes, number of [residual blocks](https://autocpd.readthedocs.io/en/latest/autocpd.html#autocpd.neuralnetwork.resblock), etc, see more arguments description [here](https://autocpd.readthedocs.io/en/latest/autocpd.html#autocpd.neuralnetwork.general_deep_nn). To call these functions, just import them into Python script as:

```python
from autocpd.neuralnetwork import general_deep_nn, general_simple_nn
```

## Examples

In this section, we demonstrate 3 examples to show how to train simple and deep neural networks, compile and train them and how to detect the change-points. Each example has a corresponding Python script which can be found in the folder [./test](https://github.com/Jieli12/AutoCPD/tree/master/test)

### Simple Neural Network

The Python script `test_simple_nn.py` consists of four parts:

* Data Generation
* Model Construction
* Model Compilation and Fitting
* Model Prediction

#### Data Generation

We set the length of time series $n=100$, the training data size $N=400$. The sample size of Class 0 (without change-point) and Class 1 (with only 1 change-point) are equal. This example considers detecting change in mean for the piecewise constant signal + Gaussian noise model. Other parameter settings can be found in our [paper](https://arxiv.org/abs/2211.03860). Here lists the Python script for generating training data set:

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

**Note:** `AutoCPD` also offers the utility functions to generate the training data set for change in variance and slope with/without Gaussian noise. For non-Gaussian noise, only Cauchy noise and AR(1) noise are available in current version, please see more functions in `autocpd.utils` [module](https://autocpd.readthedocs.io/en/latest/autocpd.html#module-autocpd.utils).

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

**Note:**  The default value of `validation_split` in `compile_and_fit()` is 0.2, which means that in this example, the validation data set contains the 80 samples.

### The result for multiple layer neural network

To obtain the result of multiple layer neural network:
$\mathcal{H}_{5,m^{(1)}\mathbf{1}_{5}}$ and $\mathcal{H}_{10,m^{(1)}\mathbf{1}_{10}}$, please run the following scripts in order:

* ScenarioS1Rho0L5Train.py;
* ScenarioARho0L5Predict.py;
* ScenarioS1Rho0L10Train.py;
* ScenarioARho0L10Predict.py;

Finally, run the following script to generate Fig. 2(a):

* ScenarioS1Rho0MultiLPredictReplot.py;

The figure is automatically saved in folder ’./Figures’. Note: You can only see the plotted figure using Ipython.

## HASC data analysis (Note: please wait for our Python package AutoCPD)

**If you still tried to run the scripts in the current repository, please be careful with the parameters settings.**

The HASC data downloaded from [here](http://hasc.jp/hc2011/index-en.html) is in folder ”./datasets/HASC/”. The architecture details of residual neural network can be found in supplementary material. The script “RealDataKS25.py” trains the model and saves the trained model in the folder ”./Code/tensorboard_logs/Trial/”.

**Note:** it costs several hours to train the residual neural network in GPU server. For convenience, I also put the trained model ``RealDataKS25’’ in folder ”./Code/tensorboard_logs/Trial/”. To obtain the Figure 4 in the main text, please run the file “RealDataCPDTestNewSeq1.py”.

## Extra Simulation for Multiple Change-types

**Please do not run these scripts as it was not updated since the first submission and wait for our Python package AutoCPD.**

In supplementary material, we also provide an extra simulation for one-change-point but with multiple change-types: change in mean, change in slope and change in variance.

For likelihood-ratio-based methods, we employ the Narrowest-Over-Threshold (NOT) (Baranowski et al., 2019) and single variance change-point detection (Chen and Gupta, 2012) algorithms to detect the change in mean, slope and variance respectively. The algorithms are available in **R** packages: [not](https://CRAN.R-project.org/package=not) and [changepoint](https://CRAN.R-project.org/package=changepoint).

The scripts “ResNetN1kE8tanhDecay10kScale.py” and “ResNetN1kE8tanhStrongDecay10kScale.py” can produce confusion matrices for weak and strong signal scenarios respectively. For convenience, the trained models are also available in ”./Code/tensorboard_logs/Trial/”.

To generate Table 1 in main text, please run the following scripts in order:

* RNWeak21T2N2500R10SaveForR.py;
* RNStrong21T2N2500R10SaveForR.py;
* RNWeakoracle-revisionTestR10.r;
* RNStrongoracle-revisionTestR10.r;

## Simulations in supplement

### Simulation for simultaneous changes

By running the following two Python scripts, we can get the results displayed in Table S2 of supplement.

* DataGenForRStrong2ClassTesting.py;
* DataGenForRWeak2ClassTesting.py;

### Simulation for heavy-tailed noise

The trained models can be found in ”./Code/tensorboard_logs/Trial/”.

By running the following two Python scripts, we can get the results displayed in Figure S1 of supplement.

* S3R0Predict.py;
* S3R0PredictPlot.py;

### Simulation for robustness study

The trained models can be found in ”./Code/tensorboard_logs/Trial/”.

By running the following two Python scripts, we can get the results displayed in Figure S2 of supplement.

* SARhoToOthersPredict.py;
* SARhoToOthersPredictReplot.py;

### Simulation for change in autocorrelation

The trained models can be found in ”./Code/tensorboard_logs/Trial/”.

By running the following four Python scripts, we can get the results displayed in Figure S3 of supplement.

* SAPredict.py;
* SASquarePredict.py;
* SASquareResNetPredict.py;
* SARPlot.py;

### Simulation on change-point location estimation

The trained models can be found in ”./Code/tensorboard_logs/Trial/”. There are four panels in Fig.S4 of supplement. The top two panels can be generated by running the following scripts:

* DS1WeakResult.py; # Note: it will cost serval hours.
* DS1StrongResult.py; # Note: it will cost serval hours.
* DS1WeakResultMosumAdjust.py;
* DS1StrongResultMosumAdjust.py;
* DS1ReportWilcox.py;

Similarly, the bottom two panels can be generated by running the following scripts:

* DS3WeakResult.py; # Note: it will cost serval hours.
* DS3StrongResult.py; # Note: it will cost serval hours.
* DS3WeakResultMosumAdjust.py;
* DS3StrongResultMosumAdjust.py;
* DS3WeakResultWilcoxR.py;
* DS3StrongResultWilcoxR.py;
* DS3WeakWilcox.r;
* DS3StrongWilcox.r;
* DS3ReportWilcox.py;
