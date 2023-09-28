import tensorflow as tf


def load_pretrained_model(path):
    """Load the pretrained model

    Parameters
    ----------
    path : str
        the path of pre-trained model

    Returns
    -------
    tf.Model
        the pre-trained model.
    """
    model = tf.keras.models.load_model(path)
    return model
