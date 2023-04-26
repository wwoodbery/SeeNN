import tensorflow as tf
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers import PReLU, LeakyReLU
from keras.layers import add
from keras.layers.convolutional import UpSampling2D
from keras.layers import Flatten


def discriminator_block(model, filters, kernel_size, strides):
    model = Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = "same")(model)
    model = BatchNormalization(momentum = 0.5)(model)
    model = LeakyReLU(alpha = 0.2)(model)
        
    return model

class Discriminator(object):

    def __init__(self, image_shape):
        self.image_shape = image_shape

    def discriminator(self):
        
        # This should be (360, 360, 3)
        inputs = Input(shape = self.image_shape)
        
        conv = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = "same")(inputs)
        disc_inp = LeakyReLU(alpha = 0.2)(conv)

        disc_out = discriminator_block(disc_inp, 64, 3, 2)
        disc_out = discriminator_block(disc_out, 128, 3, 1)
        disc_out = discriminator_block(disc_out, 128, 3, 2)
        disc_out = discriminator_block(disc_out, 256, 3, 1)
        disc_out = discriminator_block(disc_out, 256, 3, 2)
        disc_out = discriminator_block(disc_out, 512, 3, 1)
        disc_out = discriminator_block(disc_out, 512, 3, 2)

        flattened = Flatten()(disc_out)
        dense1 = Dense(1024)(flattened)
        leaky = LeakyReLU(alpha = 0.2)(dense1)
       
        dense2 = Dense(1, activation='sigmoid')(leaky)
        
        discriminator_model = Model(inputs = inputs, outputs = dense2)
        
        return discriminator_model
        
        

        