"""
Author         : Jie Li, Department of Statistics, London School of Economics.
Date           : 2023-09-14 18:57:05
Last Revision  : 2023-09-19 11:53:21
Last Author    : Jie Li
File Path      : /AutoCPD/testUtils.py
Description    :








Copyright (c) 2023 by Jie Li, j.li196@lse.ac.uk
All Rights Reserved.
"""
# %%
import pathlib
from collections import Counter

# %%
from itertools import groupby
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
from scipy.signal import find_peaks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sympy import diff
from utils import *

# %%
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
    "person107",
]
# %% load the trained model
logdir = Path("tensorboard_logs", "Trial")
model_name = "RealDataKS25"
model_path = Path(logdir, model_name, "model")
model = tf.keras.models.load_model(model_path)
model.summary()
label_dict = np.load(Path(logdir, model_name, "label_dict.npy"), allow_pickle=True)
label_dict = label_dict.item()
# %%  load the sequences from subject 7, data preprocessing and plot with CPs

# load the sequences, there are 3 sequences.
subject = subjects[6]
subject_path = datapath + subject
# get the csv files
all_files = os.listdir(subject_path)
csv_files = list(filter(lambda f: f.endswith(".csv"), all_files))
csv_files = list(filter(lambda f: f.startswith("HASC"), csv_files))
sequences_list = list()
cp_list = list()
label_list = list()
