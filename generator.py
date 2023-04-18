import tensorflow as tf
from keras.layers import Dense
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers import PReLU

def residual_block(kernel_size, filters, strides):
    return \
    Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding="same"), \
    BatchNormalization(momentum=0.5), \
    PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2]),\
    Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding="same"),\
    BatchNormalization(momentum=0.5)

class Generator(object):

    def generator(self, input):

        model = tf.keras.Sequential([
            Conv2D(filters=64, kernel_size=9, strides=1, padding="same"),
            PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2]),
            residual_block(3,64,1),
            residual_block(3,64,1),
            residual_block(3,64,1),
            residual_block(3,64,1),
            residual_block(3,64,1),
            residual_block(3,64,1),
            residual_block(3,64,1),
            residual_block(3,64,1),
            residual_block(3,64,1),
            residual_block(3,64,1),
            residual_block(3,64,1),
            residual_block(3,64,1),
            residual_block(3,64,1),
            residual_block(3,64,1),
            residual_block(3,64,1),
            residual_block(3,64,1),
            Conv2D(filters=64, kernel_size=3, strides=1, padding="same"),
            BatchNormalization(momentum = 0.5)(model)
        ])