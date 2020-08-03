import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import  matplotlib.pyplot as plt
import  numpy as np
import  glob
import os
from PIL import Image
import glob

def load_image(filename):
    images = os.listdir(filename)
    x_train = np.empty((images.__len__(),64,64,3),dtype='float32')

    for i in range(len(images)):
        image = Image.open(filename + images[i])
        image = image.resize((64,64))
        image_arr = np.asarray(image,dtype='float32')
        x_train[i,:,:,:] = image_arr/ 127.5 - 1.

    return  x_train

class DCGAN():
    def __init__(self):
        self.image_row = 64
        self.image_col = 64
        self.channels = 3
        self.image_shape = (self.image_row,self.image_col,self.channels)

        self.noise_dim = 100
        #optimizer
        optimizer = tf.keras.optimizers.Adam(0.0002,0.5)
        self.discriminator = self.discriminator_model()
        self.discriminator.compile(loss=['binary_crossentropy'],optimizer=optimizer,metrics=['accuracy'])

        self.generator = self.generator_model()
        z = layers.Input(shape=(self.noise_dim,))
        image = self.generator(z)
        self.discriminator.trainable = False
        val = self.discriminator(image)

        self.combined = tf.keras.models.Model(z,val)
        self.combined.compile(loss='binary_crossentropy',optimizer=optimizer)

    def generator_model(self):
        model = tf.keras.Sequential()

        # 8,8,256   16,16,256
        model.add(layers.Dense(256*64,activation='relu',input_dim=self.noise_dim))
        model.add(layers.Reshape((8,8,256)))
        model.add(layers.UpSampling2D())

        #16,16,256  32,32,128
        model.add(layers.Conv2D (128, kernel_size=5, padding='same', use_bias=False))
        model.add(layers.BatchNormalization(momentum=0.8))
        model.add(layers.Activation('relu'))
        model.add(layers.UpSampling2D())

        #32,32,64 64,64,64
        model.add(layers.Conv2D(64,kernel_size=5,padding='same',use_bias=False))
        model.add(layers.BatchNormalization(momentum=0.8))
        model.add(layers.Activation('relu'))
        model.add(layers.UpSampling2D())

        #64,64,3
        model.add(layers.Conv2D(3,kernel_size=5,padding='same'))
        model.add(layers.Activation('tanh'))

        noise = layers.Input(shape = (self.noise_dim,))
        image = model(noise)

        return tf.keras.models.Model(noise,image)

    def discriminator_model(self):
        model = tf.keras.Sequential()

        #32,32,64
        model.add(layers.Conv2D(64,kernel_size=5, strides=2,input_shape=self.image_shape,padding='same'))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Dropout(0.5))

        #16,16,128
        model.add(layers.Conv2D(128,kernel_size=5,strides=2,padding='same'))
        model.add(layers.BatchNormalization(momentum=0.8))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Dropout(0.5))

        #8,8,256
        model.add(layers.Conv2D(256,kernel_size=5,strides=2,padding='same'))
        model.add(layers.BatchNormalization(momentum=0.8))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Dropout(0.5))

        model.add(layers.GlobalAveragePooling2D())
        model.add(layers.Dense(1,activation='sigmoid'))

        image = layers.Input(shape=self.image_shape)
        val = model(image)

        return tf.keras.models.Model(image,val)

    def save_image(self,epoch):
        x = 3
        y = 3
        noise = np.random.normal(0,1,(x * y,self.noise_dim))
        gen_image = self.generator.predict(noise)
        gen_image = 127.5 * gen_image + 127.5

        fig,axs = plt.subplots(x,y)
        cnt = 0
        for i in range(x):
            for j in range(y):
                gen_image = gen_image.astype(np.uint8)
                axs[i,j].imshow(gen_image[cnt,:,:,:])
                axs[i,j].axis('off')
                cnt +=1
        fig.savefig("GANANIME_Images4/DCGANFaces_%d.png" % epoch)
        plt.close()


    def train(self ,epochs , batch_size = 128, save_interval = 100):

        filename = "faces/"
        x_train = load_image(filename)

        val = np.ones((batch_size,1))
        fake = np.zeros((batch_size,1))

        for epochs in range(epochs):
            idx = np.random.randint(0,x_train.shape[0],batch_size)
            image = x_train[idx]

            noise = np.random.normal(0,1,(batch_size,self.noise_dim))
            gen_image = self.generator.predict(noise)

            real_loss = self.discriminator.train_on_batch(image,val)
            fake_loss = self.discriminator.train_on_batch(gen_image,fake)
            disc_loss = 0.5 * np.add(real_loss,fake_loss)

            gen_loss = self.combined.train_on_batch(noise,val)

            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epochs, disc_loss[0], 100*disc_loss[1], gen_loss))

            if epochs % save_interval ==0 :
                self.save_image(epochs)



if __name__ =='__main__':
    if not os.path.exists("./GANANIME_Images4"):
        os.makedirs("./GANANIME_Images4")
    dcgan = DCGAN()
    dcgan.train(epochs=10000)


