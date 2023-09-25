"""
Author         : Jie Li, Department of Statistics, London School of Economics.
Date           : 2023-09-25 10:56:59
Last Revision  : 2023-09-25 11:06:25
Last Author    : Jie Li
File Path      : /AutoCPD/src/autocpd/pre_trained_model.py
Description    :








Copyright (c) 2023 by Jie Li, j.li196@lse.ac.uk
All Rights Reserved.
import
"""
import tensorflow as tf


def load_pretrained_model():
    """
    Load the pretrained model

    Returns
    -------
    tf.Model
        the pre-trained model.
    """
    model = tf.keras.models.load_model("Demo/model")
    return model
