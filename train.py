import tensorflow as tf
import numpy as np
from keras.applications.vgg19 import VGG19
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input
from generator import Generator
from discriminator import Discriminator
import preprocess
import pickle
import time


image_shape = preprocess.get_hr_shape()

start = time.time()
file = open('lr_hr_image_data2.pkl','rb')
data = pickle.load(file)

print(data[0][1].shape)

lr_images = []
hr_images = []

indices = np.arange(len(data))
np.random.shuffle(indices)

for indice in indices:
  lr_images.append(data[indice][0])
  hr_images.append(data[indice][1])

end = time.time()
print(end-start)

# check to see if there are any images in hr_images that do not have shape (358, 358, 3)
# for i in range(len(hr_images)):
#   if hr_images[i].shape != (358, 358, 3):
#     print(i)

print(hr_images[0].shape)
print(lr_images[0].shape)


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    results = []
    for i in range(0, len(lst), n):
        results.append(lst[i:i + n])
      
    return list(results)

def vgg_loss(y_true, y_pred):
    vgg19 = VGG19(include_top=False, weights='imagenet', input_shape=image_shape)
    vgg19.trainable = False
    for l in vgg19.layers:
        l.trainable = False
    loss_model = Model(inputs=vgg19.input, outputs=vgg19.get_layer('block5_conv4').output)
    loss_model.trainable = False
    mse = tf.reduce_mean(tf.math.squared_difference(loss_model(y_true), loss_model(y_pred)))
    return mse

def get_gan_network(discriminator, shape, generator, optimizer):
    discriminator.trainable = False
    gan_input = Input(shape=shape)
    gen_out = generator(gan_input)
    gan_output = discriminator(gen_out)
    gan = Model(inputs=gan_input, outputs=[gen_out,gan_output])
    gan.compile(loss=[vgg_loss, "binary_crossentropy"],
                loss_weights=[1., 1e-3],
                optimizer=optimizer)
    return gan

def train(epochs=1, batch_size=1):
    downscale_factor = 4
    #batch_count = None
    shape = (image_shape[0]//downscale_factor, image_shape[1]//downscale_factor, image_shape[2])
    
    generator = Generator(shape).generator()
    discriminator = Discriminator(image_shape).discriminator()

    adam = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    generator.compile(loss=vgg_loss, optimizer=adam)
    discriminator.compile(loss="binary_crossentropy", optimizer=adam)

    gan = get_gan_network(discriminator, shape, generator, adam)

    lr_images_batches = chunks(lr_images, batch_size)
    hr_images_batches = chunks(hr_images, batch_size)

    for e in range(1, epochs+1):
        print ('-'*15, 'Epoch %d' % e, '-'*15)
        for batch in range(batch_size):
            print("Batch #", batch)
            image_batch_hr = tf.convert_to_tensor(hr_images_batches[batch])
            image_batch_lr = tf.convert_to_tensor(lr_images_batches[batch])
            print("generating sr images")
            generated_images_sr = generator.predict(image_batch_lr)
            print("done generating sr images")
            real_data_Y = np.ones(batch_size)
            fake_data_Y = np.zeros(batch_size)

            discriminator.trainable = True

            print("d_loss_real start")
            d_loss_real = discriminator.train_on_batch(image_batch_hr, real_data_Y)
            print("d_loss_real end")
            d_loss_fake = discriminator.train_on_batch(generated_images_sr, fake_data_Y)

            gan_Y = np.ones(batch_size)

            discriminator.trainable = False
            
            loss_gan = gan.train_on_batch(image_batch_lr, [image_batch_hr, gan_Y])

        print("Loss HR , Loss LR, Loss GAN")
        print(d_loss_real, d_loss_fake, loss_gan)

        if e == 1 or e % 5 == 0:
            pass
            #plot_generated_images(e, generator)
        if e % 300 == 0:
            generator.save('./output/gen_model%d.h5' % e)
            discriminator.save('./output/dis_model%d.h5' % e)
            gan.save('./output/gan_model%d.h5' % e)

train(4, 1)