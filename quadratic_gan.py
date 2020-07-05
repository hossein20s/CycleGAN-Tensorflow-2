import io
import logging
import logging.config
import os
import time

import numpy
import tensorflow
from matplotlib import animation
from matplotlib import pyplot
from tensorflow.keras import layers

from utils import make_animation, generator_loss, discriminator_loss

SCALE = 10
noise_dim = 100

class QuadraticGAN:

    def __init__(self, target_function, initial_batch_size, number_of_batch, number_of_interation, epochs) -> None:
        super().__init__()
        logging.config.fileConfig(fname='log.conf')
        self.logger = logging.getLogger('dev')
        self.epochs = epochs
        self.target_function = target_function
        self.initial_batch_size = initial_batch_size
        self.number_of_interation = number_of_interation
        self.number_of_batch = number_of_batch
        self.generator = self.make_generator_model()
        self.discriminator = self.make_discriminator_model()
        self.generator_optimizer = tensorflow.keras.optimizers.Adam(1e-4)
        self.discriminator_optimizer = tensorflow.keras.optimizers.Adam(1e-4)
        self.checkpoint = tensorflow.train.Checkpoint(step=tensorflow.Variable(1),
                                                      generator_optimizer=self.generator_optimizer,
                                                      discriminator_optimizer=self.discriminator_optimizer,
                                                      generator=self.generator,
                                                      discriminator=self.discriminator)
        self.manager = tensorflow.train.CheckpointManager(self.checkpoint, '/opt/host/gdocs/checkpoints/quadraticGAN', max_to_keep=3)
        self.all_image_buffers = []


    def run(self, debug=False):
        batch_size = self.initial_batch_size
        for _ in range(self.number_of_interation):
            self.run_iteration(batch_size=batch_size, epochs=self.epochs, debug=debug)
            batch_size *= 2
        if self.all_image_buffers:
            make_animation(self.all_image_buffers, '/opt/host/Downloads/quadGAN')

    def run_iteration(self, batch_size, epochs, debug=False):
        number_of_sample = self.number_of_batch * batch_size
        # data = self.sample_data(number_of_sample, scale=SCALE)
        data = self.target_function()
        image_buffers = self.train(data=data, epochs=epochs, number_of_sample=number_of_sample, debug=debug)
        self.all_image_buffers.extend(self.make_animation(image_buffers))

    def train_step(self, x_batch):
        noise = tensorflow.random.normal([len(x_batch), noise_dim])

        with tensorflow.GradientTape() as gen_tape, tensorflow.GradientTape() as disc_tape:
            generated_data = self.generator(noise, training=True)
            real_output = self.discriminator(x_batch, training=True)
            fake_output = self.discriminator(generated_data, training=True)

            gen_loss = generator_loss(fake_output)
            disc_loss = discriminator_loss(real_output, fake_output)

            gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, self.discriminator.trainable_variables))

    def train(self, data, epochs, number_of_sample, debug=False):
        self.checkpoint.restore(self.manager.latest_checkpoint)
        if self.manager.latest_checkpoint:
            self.logger.info("Restored from {}".format(self.manager.latest_checkpoint))
        else:
            self.logger.info("Initializing from scratch.")

        dataset = tensorflow.data.Dataset.from_tensor_slices(data).batch(self.initial_batch_size)
        test_input = tensorflow.random.normal([number_of_sample, noise_dim])
        start = time.time()
        image_buffers = []
        log_period = min(10, epochs // 10)
        snapshot_period = min(10, (epochs // 50) + 1)
        self.logger.info('snapshotting every {} and reporting every {} epochs'.format(snapshot_period, log_period))
        for epoch in range(epochs):
            print(epoch, sep=' ', end='', flush=True)
            for x_batch in dataset:
                self.train_step(x_batch)
            # self.logger.info(epoch)

            if epoch % log_period == 0:
                self.logger.info('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))
                # f.write("%d,%f,%f\n" % (i, dloss, gloss))

            if epoch % snapshot_period == 0:
                # Produce images for the GIF as we go
                # display.clear_output(wait=True)
                image_buffers.append(
                    self.log_loss_and_save_images(epoch=epoch + 1, data=data, test_input=test_input, debug=debug))
            self.checkpoint.step.assign_add(1)
            if int(self.checkpoint.step) % 10 == 0:
                save_path = self.manager.save()
                self.logger.info("Saved checkpoint for step {}: {}".format(int(self.checkpoint.step), save_path))

        # Generate after the final epoch
        # display.clear_output(wait=True)
        image_buffers.append(self.log_loss_and_save_images(data=data, epoch=epochs, test_input=test_input, debug=debug))
        return image_buffers

    def make_animation(self, image_buffers):
        # First set up the figure, the axis, and the plot element we want to animate
        fig, ax = pyplot.subplots(nrows=1, ncols=1, figsize=(10, 10))
        pyplot.close()
        ax.xlim = (0, 10)
        ax.ylim = (0, 10)
        ax.set_xticks([])
        ax.set_yticks([])
        buffer = image_buffers[0]
        buffer.seek(0)
        img = ax.imshow(pyplot.imread(buffer))
        img.set_interpolation('nearest')

        def updatefig(frame, buffers):
            buffer = buffers[frame]
            buffer.seek(0)
            img.set_data(pyplot.imread(buffer))
            return (img,)

        animation_function = animation.FuncAnimation(fig, updatefig, frames=len(image_buffers), fargs=(image_buffers,),
                                                     interval=100, blit=True)
        anim_writer = animation.writers['ffmpeg']
        writer = anim_writer(fps=15, metadata=dict(artist='Hossein'), bitrate=1800)
        time_string = time.strftime("%Y%m%d-%H%M%S")
        file_name = '{}{}-{}.mp4'.format('/opt/host/Downloads/', QuadraticGAN.__name__, time_string)
        animation_function.save(file_name, writer=writer, )
        return image_buffers

    def log_loss_and_save_images(self, data, epoch, test_input, debug=False):
        # Notice `training` is set to False.
        # This is so all layers run in inference mode (batchnorm).
        generated_data = self.generator(test_input, training=False)
        real_output = self.discriminator(data, training=False)
        fake_output = self.discriminator(generated_data, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
        self.logger.info(
            'epoch {}, gen loss {}, discriminator loss {}'.format(epoch, gen_loss.numpy(), disc_loss.numpy()))
        pyplot.close()
        pyplot.scatter(*zip(*generated_data.numpy()))
        # ax = pyplot.gca()
        # pyplot.scatter(*zip(*data))
        pyplot.scatter(data[0], data[1])
        # ax = data.plot(kind='scatter', x=0, y=1)
        # fig = ax.get_figure()

        buffer = io.BytesIO()
        pyplot.savefig(buffer, format='png')
        if debug:
            buffer.seek(0)
            pyplot.imread(buffer)
            pyplot.show()
        return buffer

    def sample_data(self, number_of_sample, scale):
        data = []
        x = (numpy.random.random((number_of_sample,)) - .5) * scale
        for i in range(number_of_sample):
            data.append((x[i], self.target_function(x[i])))
        return numpy.array(data)

    @staticmethod
    def make_generator_model():
        model = tensorflow.keras.Sequential()
        model.add(layers.Dense(16, use_bias=False, input_shape=(100,)))
        model.add(layers.LeakyReLU())
        model.add(layers.Dense(16, use_bias=False))
        model.add(layers.LeakyReLU())
        model.add(layers.Dense(2, use_bias=False))
        return model

    @staticmethod
    def make_discriminator_model():
        model = tensorflow.keras.Sequential()
        model.add(layers.Dense(16, use_bias=False, input_shape=(2,)))
        model.add(layers.LeakyReLU())
        model.add(layers.Dense(16, use_bias=False))
        model.add(layers.LeakyReLU())
        model.add(layers.Dense(2, use_bias=False))
        model.add(layers.LeakyReLU())
        model.add(layers.Dense(2, use_bias=False))
        model.add(layers.Dense(1, use_bias=False))
        return model

