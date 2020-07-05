import os
import time

import numpy
import seaborn
import tensorflow
from keras.datasets import mnist
from matplotlib import pyplot
from tensorflow.keras import layers

from utils import generator_loss, discriminator_loss

seaborn.set()

BUFFER_SIZE = 60000
BATCH_SIZE = 256
EPOCHS = 50
noise_dim = 100
num_examples_to_generate = 16
checkpoint_dir = '/opt/host/Downloads/training_checkpoints'

# We will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tensorflow.random.normal([num_examples_to_generate, noise_dim])


class MnistGAN:

    def __init__(self) -> None:
        super().__init__()
        self.generator = self.make_generator_model()
        self.discriminator = self.make_discriminator_model()
        self.generator_optimizer = tensorflow.keras.optimizers.Adam(1e-4)
        self.discriminator_optimizer = tensorflow.keras.optimizers.Adam(1e-4)

        self.checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        self.checkpoint = tensorflow.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                                      discriminator_optimizer=self.discriminator_optimizer,
                                                      generator=self.generator,
                                                      discriminator=self.discriminator)
        (self.train_images, self.train_labels), (_, _) = tensorflow.keras.datasets.mnist.load_data()

    # Notice the use of `tensorflow.function`
    # This annotation causes the function to be "compiled".
    @tensorflow.function
    def train_step(self, images):
        noise = tensorflow.random.normal([BATCH_SIZE, noise_dim])

        with tensorflow.GradientTape() as gen_tape, tensorflow.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)

            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            gen_loss = generator_loss(fake_output)
            disc_loss = discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, self.discriminator.trainable_variables))

    def train(self, dataset, epochs):
        for epoch in range(epochs):
            start = time.time()

            for image_batch in dataset:
                self.train_step(image_batch)

            # Produce images for the GIF as we go
            # display.clear_output(wait=True)
            self.generate_and_save_images(self.generator, epoch + 1, seed)
            # Save the model every 15 epochs
            if (epoch + 1) % 15 == 0:
                self.checkpoint.save(file_prefix=self.checkpoint_prefix)
            print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))

        # Generate after the final epoch
        # display.clear_output(wait=True)
        self.generate_and_save_images(self.generator, epochs, seed)

    def run(self):
        # Batch and shuffle the data
        train_images = self.train_images.reshape(self.train_images.shape[0], 28, 28, 1).astype('float32')
        train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]
        train_dataset = tensorflow.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

        self.train(train_dataset, EPOCHS)
        self.checkpoint.restore(tensorflow.train.latest_checkpoint(checkpoint_dir))

    @staticmethod
    def make_generator_model():
        model = tensorflow.keras.Sequential()
        model.add(layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(100,)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Reshape((7, 7, 256)))
        assert model.output_shape == (None, 7, 7, 256)  # Note: None is the batch size

        model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
        assert model.output_shape == (None, 7, 7, 128)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, 14, 14, 64)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
        assert model.output_shape == (None, 28, 28, 1)

        return model


    @staticmethod
    def make_discriminator_model():
        model = tensorflow.keras.Sequential()
        model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                input_shape=[28, 28, 1]))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Flatten())
        model.add(layers.Dense(1))

        return model

    def train1(self, epochs, batch_size=128, save_interval=5):
        # Load the dataset
        (x_train, _), (_, _) = mnist.load_data()

        # Rescale -1 to 1
        x_train = (x_train.astype(numpy.float32) - 127.5) / 127.5
        x_train = numpy.expand_dims(x_train, axis=3)

        half_batch = int(batch_size / 2)

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            idx = numpy.random.randint(0, x_train.shape[0], half_batch)
            imgs = x_train[idx]

            noise = numpy.random.normal(0, 1, (half_batch, 100))

            # Generate a half batch of new images
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, numpy.ones((half_batch, 1)))
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, numpy.zeros((half_batch, 1)))
            d_loss = 0.5 * numpy.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            noise = numpy.random.normal(0, 1, (batch_size, 100))

            # The generator wants the discriminator to label the generated samples
            # as valid (ones)
            valid_y = numpy.array([1] * batch_size)

            # Train the generator
            g_loss = self.combined.train_on_batch(noise, valid_y)

            # Plot the progress
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch)

    @staticmethod
    def generate_and_save_images(model, epoch, test_input):
        # Notice `training` is set to False.
        # This is so all layers run in inference mode (batchnorm).
        predictions = model(test_input, training=False)

        pyplot.figure(figsize=(4, 4))

        for i in range(predictions.shape[0]):
            pyplot.subplot(4, 4, i + 1)
            pyplot.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
            pyplot.axis('off')

        pyplot.savefig('image_at_epoch_{:04d}.png'.format(epoch))
        pyplot.show()
