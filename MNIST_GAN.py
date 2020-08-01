import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import  matplotlib.pyplot as plt
import  numpy as np
import  glob
import os



(train_images,train_labels),(_,_) = tf.keras.datasets.mnist.load_data()
# reshape real image
train_images = train_images.reshape(train_images.shape[0],28,28,1).astype('float32')
train_images = (train_images - 127.5)/127.5

BATCH_SIZE = 256
BUFFER_SIZE = 60000

datasets = tf.data.Dataset.from_tensor_slices(train_images)
datasets = datasets.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

def generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*256,activation='relu',input_shape = (100,),use_bias=False))
    model.add(layers.Reshape((7,7,256)))

    model.add(layers.Conv2D(128,kernel_size=5,padding='same',use_bias=False))
    #model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    #model.add(layers.Activation('relu'))
    model.add(layers.UpSampling2D())

    model.add(layers.Conv2D(64,kernel_size=5,padding='same',use_bias=False))
    #model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.UpSampling2D())
    #imput (28,28,64)

    model.add(layers.Conv2D(1,kernel_size=5,padding='same' ,use_bias=False))
    model.add(layers.Activation('tanh'))

    model.add(layers.Reshape((28, 28, 1)))
    return model

def discrimator_model():
    model = tf.keras.Sequential()

    model.add(layers.Conv2D(64,kernel_size=5,strides=2,input_shape=[28,28,1],padding='same'))
    model.add(layers.LeakyReLU())
    #model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128,kernel_size=5,strides=2,padding='same'))
    #model.add(layers.BatchNormalization (momentum=0.8))
    #model.add(layers.LeakyReLU(alpha=0.3))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

#loss function

def discriminator_loss(real_out, fake_out):
    real_loss = cross_entropy(tf.ones_like(real_out),real_out)
    fake_loss = cross_entropy(tf.zeros_like(fake_out),fake_out)
    return real_loss + fake_loss

def generator_loss(fake_out):
    return cross_entropy(tf.ones_like(fake_out),fake_out)

#optimization

generator_opt = tf.keras.optimizers.Adam(1e-4)
discriminator_opt = tf.keras.optimizers.Adam(1e-4)


noise_dim = 100
num_exp_to_generate = 16
seed = tf.random.normal([num_exp_to_generate,noise_dim])

#
generator = generator_model()
discriminator = discrimator_model()

#traning precess

def train_step(image):
    noise = tf.random.normal([BATCH_SIZE,noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        real_out = discriminator(image,training=True)
        gen_image = generator(noise,training =True)
        fake_out = discriminator(gen_image,training=True)

        gen_loss = generator_loss(fake_out)
        disc_loss =  discriminator_loss(real_out,fake_out)

        gradient_gen = gen_tape.gradient(gen_loss,generator.trainable_variables)
        gradient_disc = disc_tape.gradient(disc_loss,discriminator.trainable_variables)

        generator_opt.apply_gradients(zip(gradient_gen,generator.trainable_variables))
        discriminator_opt.apply_gradients(zip(gradient_disc,discriminator.trainable_variables))

def generate_plot_image(gen_model,test_noise):
    pre_images = gen_model(test_noise,training = False)
    fig = plt.figure(figsize=(4,4))
    for i in range(pre_images.shape[0]):
        plt.subplot(4,4,i+1)
        plt.imshow((pre_images[i,:,:,0] + 1) * 127.5 , cmap ='gray')
        plt.axis('off')
    plt.show()

EPOCHS = 50

def train(dataset , epochs):
    for epoch in range(epochs):
        for image_batch in dataset:
            train_step(image_batch)

        generate_plot_image(generator,seed)
    print(epochs)



train(datasets,EPOCHS)