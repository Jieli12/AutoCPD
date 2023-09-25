"""
Author         : Jie Li, Department of Statistics, London School of Economics.
Date           : 2023-09-25 11:11:18
Last Revision  : 2023-09-25 18:21:53
Last Author    : Jie Li
File Path      : /AutoCPD/test/test_load_pretrained_model.py
Description    :








Copyright (c) 2023 by Jie Li, j.li196@lse.ac.uk
All Rights Reserved.
"""
import os
import pathlib

import autocpd
from autocpd.pre_trained_model import load_pretrained_model

root_path = os.path.dirname(autocpd.__file__)
model_path = pathlib.Path(root_path, "Demo", "model")

model = load_pretrained_model(model_path)
