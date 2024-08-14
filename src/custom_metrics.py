import tensorflow as tf
import keras
from keras.utils import register_keras_serializable
from keras.metrics import RootMeanSquaredError

@register_keras_serializable()
def custom_loss(y_true, y_pred):
    penalty_factor_1 = 1.1
    penalty_factor_2 = 1.6
    penalty_factor_3 = 9

    mse = tf.square(y_true - y_pred)
    penalized_mse = tf.where((y_true > 0) & (y_true <= 10), penalty_factor_1 * mse, mse)
    penalized_mse = tf.where((y_true > 70) & (y_true <= 80), penalty_factor_2 * mse, mse)
    penalized_mse = tf.where((y_true > 80) & (y_true <= 100), penalty_factor_3 * mse, mse)
    weighted_error = tf.reduce_mean(penalized_mse)
    return tf.sqrt(weighted_error)
