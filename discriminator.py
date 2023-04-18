import tensorflow as tf
from keras.models import Model


def discriminator_block(model, filters, kernel_size, strides):
    model = tf.nn.Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = "same")(model)
    model = tf.keras.layers.BatchNormalization(momentum = 0.5)(model)
    model = tf.keras.activations.LeakyReLU(alpha = 0.2)(model)
        
    return model

class Discriminator(object):

    def __init__(self, image_shape):
        self.image_shape = image_shape

    def discriminator(self, input):
        conv = tf.nn.Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = "same")(input)
        disc_inp = tf.keras.activations.LeakyReLU(alpha = 0.2)(conv)

        disc_out = discriminator_block(disc_inp, 64, 3, 2)
        disc_out = discriminator_block(disc_out, 128, 3, 1)
        disc_out = discriminator_block(disc_out, 128, 3, 2)
        disc_out = discriminator_block(disc_out, 256, 3, 1)
        disc_out = discriminator_block(disc_out, 256, 3, 2)
        disc_out = discriminator_block(disc_out, 512, 3, 1)
        disc_out = discriminator_block(disc_out, 512, 3, 2)

        flattened = tf.keras.layers.Flatten()(disc_out)
        dense1 = tf.keras.layers.Dense(1024)(flattened)
        leaky = tf.keras.activations.LeakyReLU(alpha = 0.2)(dense1)
       
        dense2 = tf.keras.layers.Dense(1, activation='sigmoid')(leaky)
        
        discriminator_model = Model(inputs = input, outputs = dense2)
        
        return discriminator_model
        
        

        