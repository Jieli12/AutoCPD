import warnings
from pathlib import Path

import numpy as np
import tensorflow as tf
from keras import layers, losses, metrics, models


def general_simple_nn(n, l, m, num_classes, model_name="simple_nn"):
    """
    To construct a simple neural network.

    Parameters
    ----------
    n : scalar
        the input size
    l : scalar
        the number of hidden layers
    m : scalar or 1D array
        the width vector of hidden layers, if it is a scalar, then the hidden layers of simple neural network have the same nodes.
    num_classes : scalar
        the nodes of output layers, i.e., the number of classes
    model_name : str, optional
        the model name, by default "simple_nn"

    Returns
    -------
    model
        the simple neural network
    """
    input_layer = layers.Input(shape=(n,), name="Input")
    if isinstance(m, int):
        m_vec = np.repeat(m, l)
    elif len(m) == l:
        m_vec = m
    else:
        warnings.warn(
            "The length of width vector must be equal to the number of hidden layers.",
            DeprecationWarning,
        )

    x = layers.Dense(m_vec[0], activation="relu", kernel_regularizer="l2")(input_layer)
    if l >= 2:
        for k in range(l - 1):
            x = layers.Dense(m_vec[k + 1], activation="relu", kernel_regularizer="l2")(
                x
            )

    output_layer = layers.Dense(num_classes)(x)
    model = models.Model(input_layer, output_layer, name=model_name)
    return model


# mymodel = simple_nn(n=100, l=1, m=10, num_classes=2)
# mymodel = simple_nn(n=100, l=3, m=10, num_classes=2)
# mymodel = simple_nn(n=100, l=3, m=[20, 20, 5], num_classes=2)

# build the model, train and save it to disk


def get_optimizer(learning_rate):
    """
    To get the optimizer given the learning rate.

    Parameters
    ----------
    learning_rate : float
        the learning rate for inverse time decay schedule.

    Returns
    -------
    optimizer
        the Adam
    """
    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
        learning_rate, decay_steps=10000, decay_rate=1, staircase=False
    )
    return tf.keras.optimizers.Adam(lr_schedule)


def get_callbacks(name, log_dir, epochdots):
    """
    Get callbacks. This function returns the result of epochs during training, if it satisfies some conditions then the training can stop early. At meanwhile, this function also save the results of training in TensorBoard and csv files.

    Parameters
    ----------
    name : str
        the model name
    log_dir : str
        the path of log files
    epochdots : object
        the EpochDots object from tensorflow_docs

    Returns
    -------
    list
        the list of callbacks
    """
    name1 = name + "/log.csv"
    return [
        epochdots,
        tf.keras.callbacks.EarlyStopping(
            monitor="val_sparse_categorical_crossentropy", patience=800, min_delta=1e-3
        ),
        tf.keras.callbacks.TensorBoard(Path(log_dir, name)),
        tf.keras.callbacks.CSVLogger(Path(log_dir, name1)),
    ]


def compile_and_fit(
    model,
    x_train,
    y_train,
    batch_size,
    lr,
    name,
    log_dir,
    epochdots,
    optimizer=None,
    validation_split=0.2,
    max_epochs=10000,
):
    """
    To compile and fit the model

    Parameters
    ----------
    model : Models object
        the simple neural network
    x_train : tf.Tensor
        the tensor of training data
    y_train : tf.Tensor
        the tensor of training data, label
    batch_size : int
        the batch size
    lr : float
        the learning rate
    name : str
        the model name
    log_dir : str
        the path of log files
    epochdots : object
        the EpochDots object from tensorflow_docs
    optimizer : optimizer object or str, optional
        the optimizer, by default None
    max_epochs : int, optional
        the maximum number of epochs, by default 10000

    Returns
    -------
    model.fit object
        a fitted model object
    """
    if optimizer is None:
        optimizer = get_optimizer(lr)
    model.compile(
        optimizer=optimizer,
        loss=losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            metrics.SparseCategoricalCrossentropy(
                from_logits=True, name="sparse_categorical_crossentropy"
            ),
            "accuracy",
        ],
    )
    history = model.fit(
        x_train,
        y_train,
        epochs=max_epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=get_callbacks(name, log_dir, epochdots),
        verbose=2,
    )
    return history


def resblock(x, kernel_size, filters, strides=1):
    """
    This function constructs a resblock.

    Parameters
    ----------
    x : tensor
        the input data
    kernel_size : int
        the kernel size
    filters : int
        the filter size
    strides : int, optional
        the stride, by default 1

    Returns
    -------
    layer
        the hidden layer
    """
    x1 = layers.Conv2D(filters, kernel_size, strides=strides, padding="same")(x)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.ReLU()(x1)
    x1 = layers.Conv2D(filters, kernel_size, padding="same")(x1)
    x1 = layers.BatchNormalization()(x1)
    if strides != 1:
        x = layers.Conv2D(filters, 1, strides=strides, padding="same")(x)
        x = layers.BatchNormalization()(x)

    x1 = layers.Add()([x, x1])
    x1 = layers.ReLU()(x1)
    return x1


def deep_nn(
    n,
    n_trans,
    kernel_size,
    n_filter,
    dropout_rate,
    n_classes,
    m,
    l,
    model_name="deep_nn",
):
    """
    This function is used to construct the deep neural network with 21 residual blocks.

    Parameters
    ----------
    n : int
        the length of time series
    n_trans : int
        the number of transformations
    kernel_size : int
        the kernel size
    n_filter : int
        the filter size
    dropout_rate : float
        the dropout rate
    n_classes : int
        the number of classes
    m : array
        the width vector
    l : int
        the number of dense layers

    model_name : str, optional
        the model name, by default "deep_nn"

    Returns
    -------
    model
        the model of deep neural network
    """
    # Note: the following network will cost several hours to train the residual neural network in GPU server.
    input_layer = layers.Input(shape=(n_trans, n), name="Input")
    x = layers.Reshape((n_trans, n, 1))(input_layer)
    x = layers.Conv2D(n_filter, 2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = resblock(x, kernel_size, filters=n_filter)
    x = resblock(x, kernel_size, filters=n_filter)
    x = resblock(x, kernel_size, filters=n_filter)

    x = resblock(x, kernel_size, filters=n_filter, strides=(1, 2))
    x = resblock(x, kernel_size, filters=n_filter)
    x = resblock(x, kernel_size, filters=n_filter)
    x = resblock(x, kernel_size, filters=n_filter)

    x = resblock(x, kernel_size, filters=n_filter, strides=(1, 2))
    x = resblock(x, kernel_size, filters=n_filter)
    x = resblock(x, kernel_size, filters=n_filter)
    x = resblock(x, kernel_size, filters=n_filter)
    x = resblock(x, kernel_size, filters=n_filter)
    x = resblock(x, kernel_size, filters=n_filter)

    x = resblock(x, kernel_size, filters=n_filter, strides=(1, 2))
    x = resblock(x, kernel_size, filters=n_filter)
    x = resblock(x, kernel_size, filters=n_filter)
    x = resblock(x, kernel_size, filters=n_filter)

    x = resblock(x, kernel_size, filters=n_filter, strides=(1, 2))
    x = resblock(x, kernel_size, filters=n_filter)
    x = resblock(x, kernel_size, filters=n_filter)
    x = resblock(x, kernel_size, filters=n_filter)

    x = layers.GlobalAveragePooling2D()(x)
    for i in range(l - 1):
        x = layers.Dense(m[i], activation="relu", kernel_regularizer="l2")(x)
        x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(m[l - 1], activation="relu", kernel_regularizer="l2")(x)
    output_layer = layers.Dense(n_classes)(x)
    model = models.Model(input_layer, output_layer, name=model_name)
    return model


def general_deep_nn(
    n,
    n_trans,
    kernel_size,
    n_filter,
    dropout_rate,
    n_classes,
    n_resblock,
    m,
    l,
    model_name="deep_nn",
):
    """
    This function is used to construct the deep neural network with 21 residual blocks.

    Parameters
    ----------
    n : int
        the length of time series
    n_trans : int
        the number of transformations
    kernel_size : int
        the kernel size
    n_filter : int
        the filter size
    dropout_rate : float
        the dropout rate
    n_classes : int
        the number of classes
    n_resnet : int
        the number of residual blocks
    m : array
        the width vector
    l : int
        the number of dense layers
    model_name : str, optional
        the model name, by default "deep_nn"

    Returns
    -------
    model
        the model of deep neural network
    """
    # Note: the following network will cost several hours to train the residual neural network in GPU server.
    input_layer = layers.Input(shape=(n_trans, n), name="Input")
    x = layers.Reshape((n_trans, n, 1))(input_layer)
    x = layers.Conv2D(n_filter, 2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    j1 = n_resblock % 4
    for _ in range(j1):
        x = resblock(x, kernel_size, filters=n_filter)
    j2 = n_resblock // 4
    if j2 > 0:
        for _ in range(j2):
            x = resblock(x, kernel_size, filters=n_filter, strides=(1, 2))
            x = resblock(x, kernel_size, filters=n_filter)
            x = resblock(x, kernel_size, filters=n_filter)
            x = resblock(x, kernel_size, filters=n_filter)

    x = layers.GlobalAveragePooling2D()(x)
    for i in range(l - 1):
        x = layers.Dense(m[i], activation="relu", kernel_regularizer="l2")(x)
        x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(m[l - 1], activation="relu", kernel_regularizer="l2")(x)
    output_layer = layers.Dense(n_classes)(x)
    model = models.Model(input_layer, output_layer, name=model_name)
    return model
