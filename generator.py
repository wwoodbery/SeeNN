import tensorflow as tf
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers import PReLU, LeakyReLU
from keras.layers import Input
from keras.layers import add
from keras.layers.convolutional import UpSampling2D
from keras.models import Model


def residual_block(inputs, kernel_size, filters, strides):
    og_inputs = inputs

    inputs = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding="same")(inputs)
    inputs = BatchNormalization(momentum=0.5)(inputs)
    inputs = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2])(inputs)
    inputs = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding="same")(inputs)
    inputs = BatchNormalization(momentum=0.5)(inputs)
    
    inputs = add([og_inputs, inputs])

    return inputs

def upsampling_block(inputs, kernel_size, filters, strides):
    inputs = Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = "same")(inputs)
    inputs = UpSampling2D(size = 2)(inputs)
    inputs = LeakyReLU(alpha = 0.2)(inputs)

    return inputs

class Generator(object):

    def __init__(self, input_shape):
        self.input_shape = input_shape

    def generator(self):

        ## storing original inputs for use in model declaration later
        og_inputs = Input(shape = self.input_shape)

        ## first block
        inputs = Conv2D(filters = 64, kernel_size = 9, strides = 1, padding = "same")(og_inputs)
        inputs = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2])(inputs)

        ## storing output of first block for skip connection later
        first_block_outputs = inputs

        ## 16 residual block
        inputs = residual_block(inputs, 3, 64, 1)
        inputs = residual_block(inputs, 3, 64, 1)
        inputs = residual_block(inputs, 3, 64, 1)
        inputs = residual_block(inputs, 3, 64, 1)
        inputs = residual_block(inputs, 3, 64, 1)
        inputs = residual_block(inputs, 3, 64, 1)
        inputs = residual_block(inputs, 3, 64, 1)
        inputs = residual_block(inputs, 3, 64, 1)
        inputs = residual_block(inputs, 3, 64, 1)
        inputs = residual_block(inputs, 3, 64, 1)
        inputs = residual_block(inputs, 3, 64, 1)
        inputs = residual_block(inputs, 3, 64, 1)
        inputs = residual_block(inputs, 3, 64, 1)
        inputs = residual_block(inputs, 3, 64, 1)
        inputs = residual_block(inputs, 3, 64, 1)
        inputs = residual_block(inputs, 3, 64, 1)
        
        ## skip connection
        inputs = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = "same")(inputs)
        inputs = BatchNormalization(momentum = 0.5)(inputs)
        inputs = add([first_block_outputs, inputs])

        ## 2 upsampling blocks
        inputs = upsampling_block(inputs, 3, 256, 1)
        inputs = upsampling_block(inputs, 3, 256, 1)

        inputs = Conv2D(filters = 3, kernel_size = 9, strides = 1, padding = "same")(inputs)
        inputs = tf.keras.activations.tanh(inputs)

        ## disambiguate naming
        model_output = inputs

        return Model(inputs=og_inputs, outputs=model_output)



