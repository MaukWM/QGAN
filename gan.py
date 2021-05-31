import numpy as np

import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.python.keras import Sequential, Input
from tensorflow.python.keras.models import Model

from data import TwoClusters, OneClusters


class GAN:

    def __init__(self):
        self.dimensions = 2

        self.latent_dim = 5

        self.data_generator = OneClusters()

        self.learning_rate = 0.0002

        optimizer = tf.keras.optimizers.Adam(self.learning_rate)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates new data points from it
        z = Input(shape=(self.latent_dim,))
        generated_data_point = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated data points as input and determines their validity
        validity = self.discriminator(generated_data_point)

        # The combined model
        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_discriminator(self):

        model = Sequential()

        model.add(tf.keras.layers.Dense(10, input_shape=(self.dimensions,)))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

        model.summary()

        disc_input = Input(self.dimensions)
        disc_output = model(disc_input)

        return Model(disc_input, disc_output)

    def build_generator(self):

        model = Sequential()

        model.add(tf.keras.layers.Dense(10, input_shape=(self.latent_dim,)))
        model.add(tf.keras.layers.Dense(10))
        model.add(tf.keras.layers.Dense(2, activation='sigmoid'))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        generated_data = model(noise)

        return Model(noise, generated_data)

    def train(self, epochs, batch_size=64, sample_interval=20):

        # Load the data, which has already been normalized
        X_train = self.data_generator.generate(100)

        plt.title("Real data points")
        plt.scatter(X_train.transpose()[0], X_train.transpose()[1])
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.savefig("plots/realdata.png")
        plt.show()

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # Train Discriminator

            # Select a random batch of real data points
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            real_data_points = X_train[idx]

            # Generate noise the the Generator
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Generate a batch of new data points
            generated_data_points = self.generator.predict(noise)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(real_data_points, valid)
            d_loss_fake = self.discriminator.train_on_batch(generated_data_points, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Train Generator

            # Generate new noise for the Generator
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Train the generator (to have the discriminator label samples as valid)
            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # Generate samples and graph them every sample_interval epochs
            if epoch % sample_interval == 0:
                self.sample_data_points(epoch)

    def sample_data_points(self, epoch):
        data_point_amount = 100
        noise = np.random.normal(0, 1, (data_point_amount, self.latent_dim))
        generated_data_points = self.generator.predict(noise)

        plt.title("Generated data points")
        plt.scatter(generated_data_points.transpose()[0], generated_data_points.transpose()[1])
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.savefig("plots/%d.png" % epoch)
        plt.show()


if __name__ == '__main__':
    gan = GAN()
    gan.train(epochs=1000, batch_size=32, sample_interval=20)
