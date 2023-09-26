"""
Author         : Jie Li, Department of Statistics, London School of Economics.
Date           : 2023-09-25 11:11:18
Last Revision  : 2023-09-26 15:36:15
Last Author    : Jie Li
File Path      : /AutoCPD/test/test_load_pretrained_model.py
Description    :








Copyright (c) 2023 by Jie Li, j.li196@lse.ac.uk
All Rights Reserved.
"""
import os
import pathlib

import numpy as np
import tensorflow as tf

import autocpd
from autocpd.pre_trained_model import load_pretrained_model

root_path = os.path.dirname(autocpd.__file__)
model_path = pathlib.Path(root_path, "Demo", "model")

model = load_pretrained_model(model_path)
model.summary()

datapath = pathlib.Path(root_path, "HASC")
np.random.seed(2022)  # numpy seed fixing
tf.random.set_seed(2022)  # tensorflow seed fixing
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
label_dict = np.load(
    pathlib.Path(root_path, "Demo", "label_dict.npy"), allow_pickle=True
)
label_dict = label_dict.item()

# %%
