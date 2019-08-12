import tensorflow as tf
import numpy as np


def l2_loss(prediction, y, config):
    # L2 Loss
    loss = tf.square(tf.subtract(y, prediction))
    loss = tf.reduce_mean(loss)

    return loss


def linearizedlie_loss(prediction, y, config):
    # Linearized geodesic loss
    # weights are computed from kinematic chain configurations and bone lengths
    weights = config.weights
    loss = tf.square(tf.subtract(y, prediction))
    loss = tf.reduce_mean(loss, axis=[0,1])
    loss = tf.reduce_mean(tf.multiply(loss, weights))

    return loss