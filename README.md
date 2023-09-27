#

<<<<<<< HEAD
## Python Environment Setting

We use TensorFlow 2.7.0 with Python 3.9.0. To save the logs and trained model of TensorFlow, we also need to install [tensorflow-docs](https://github.com/tensorflow/docs) by running

```bash
pip install git+<https://github.com/tensorflow/docs>
```

For more dependencies, please see file [tf27.yml](./tf27.yml).

## Simulation Study: S1, S2 and S3

The Section 5 in main text includes 3 scenarios: S1, S2 and S3. S1 has two cases: $\rho=0$ and $\rho=0.7$. In this section, we only illustrates how to obtain Fig. 2(a) in main text, other sub-figures can be generated by the same way. The suffixes “train” and “predict” represent the training step and prediction step respectively. The scripts with suffix “Train” will automatically save the trained model into the folder: ”./Code/tensorboard_logs/Trial/”. The scripts with suffix “Predict” will automatically load the trained model and predict the misclassification error rate. For example, to obtain the result of CUSUM, $\mathcal{H}_{1,m^{(1)}}$ and $\mathcal{H}_{1,m^{(2)}}$, where $m^{(1)} = 4\lfloor\log_2(n)\rfloor$ and $m^{(2)} = 2n-2$, you can run “ScenarioS1Rho0Train.py” and “ScenarioS1Rho0Predict.py” in order using either IPython with IDE (suggested) or Terminal.

For Ipython with IDE: just open the files, copy all the content, then paste them into Interactive Python and run.

For Terminal:

```bash
cd Code
python ScenarioS1Rho0Train.py
python ScenarioS1Rho0Predict.py
```

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
=======
This is a simple example package. You can use
[Github-flavored Markdown](https://guides.github.com/features/mastering-markdown/)
to write your content.
>>>>>>> aa8e01fd69c146615d06960aa22fd8ba32d5850d
