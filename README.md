# The Code for Automatic Change-point Detection in Time Series via Deep Learning

## Python Environment Setting

We use TensorFlow 2.7.0 with Python 3.9.0. To save the logs and trained model of TensorFlow, we also need to install ![tensorflow-docs](https://github.com/tensorflow/docs) by running

```python
pip install git+<https://github.com/tensorflow/docs>
```

For more dependencies, please see file ![tf27.yml](./tf27.yml).

## Simulation Study: S1, S2 and S3

The Section 5 in main text includes 3 scenarios: S1, S2 and S3. S1 has two cases: $\rho=0$ and $\rho=0.7$. The suffixes “train” and “predict” represent the training step and prediction step respectively. The scripts with suffex “Train” will automatically save the trained model into the folder: ”./Code/tensorboard_logs/Trial/”. The scripts with suffex “Predict” will automatically load the trained model and predict the misclassification error rate. For example, to obtain the Figure 2(a), you can run “ScenarioS1Rho0Train.py” and “ScenarioS1Rho0Predict.py” in order  using either IPython with IDE (suggested) or Terminal.

For Ipython with IDE: just open the files, copy all the content, then paste them into Interactive Python and run.

For Terminal:

```bash
cd Code
python ScenarioS1Rho0Train.py
python ScenarioS1Rho0Predict.py
```

Note: You can only see the plotted figure using Ipython.

## HASC data analysis

The HASC data downloaded from ![here](http://hasc.jp/hc2011/index-en.html) is in folder ”./datasets/HASC/”. The architecture details of residual neural network can be found in supplementary material. The script “RealDataKS25.py” train the model and save the trained model in the folder ”./Code/tensorboard_logs/Trial/”. **Note:** it costs several hours to train the residual neural network in GPU server. For convenience, I also put the trained model ``RealDataKS25’’ in folder ”./Code/tensorboard_logs/Trial/”. To obtain the Figure 4 in the main text, please run the file “RealDataCPDTestNewSeq1.py”.

## Extra Simulation for Multiple Change-types

In supplementary material, we also provide an extra simulation for one-change-point but with multiple change-types: change in mean, change in slope and change in variance.

For likelihood-ratio-based methods, we employ the Pruned Exact Linear Time (PELT) (Killick et al., 2012) and Narrowest-Over-Threshold (NOT) (Baranowski et al., 2019) algorithms to detect the change in mean, slope and variance respectively. The algorithms are available in **R** packages:![not](https://CRAN.R-project.org/package=not) and ![changepoint](https://CRAN.R-project.org/package=changepoint).

The scripts “ResNetN1kE8tanhDecay10kScale.py” and “ResNetN1kE8tanhStrongDecay10kScale.py” can produce Figures 3(c) and 3(d) in supplementary material for weak and strong signal scenarios respectively. For convenience, the trained models are also available in ”./Code/tensorboard_logs/Trial/".

For likelihood-ratio-based classifiers, please run the following scripts:
	- DataGenForRStrong.py;
	- DataGenForRWeak.py;
	- BIC-Strong.py;
	- BIC-Weak.py;
	- BICStrongPlot.py;
	- BICWeakPlot.py;
