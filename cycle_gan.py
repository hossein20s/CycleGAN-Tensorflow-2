import functools
import io
import logging
import logging.config
from argparse import Namespace

import numpy as np
import skimage
import tensorflow as tf
import tensorflow.keras as keras
import tqdm
from matplotlib import pyplot

import data
import imlib as im
import module
import pylib as py
import tf2gan as gan
import tf2lib as tl
# ==============================================================================
# =                                 train step                                 =
# ==============================================================================
from imlib import im2uint
from utils import make_animation


class CycleGAN:

    def __init__(self,
                 epochs=200,
                 epoch_decay=100,
                 pool_size=50,
                 output_dir='output',
                 datasets_dir="datasets",
                 dataset="drawing",
                 image_ext="png",
                 crop_size=256,
                 load_size=286,
                 batch_size=0,
                 adversarial_loss_mode="wgan",  # ['gan', 'hinge_v1', 'hinge_v2', 'lsgan', 'wgan']
                 lr=0.0002,
                 gradient_penalty_mode='none',  # ['none', 'dragan', 'wgan-gp'])
                 gradient_penalty_weight=10.0,
                 cycle_loss_weight=0.0,
                 identity_loss_weight=0.0,
                 beta_1=0.5,
                 color_depth=1,
                 progrssive=False):
        logging.config.fileConfig(fname='log.conf')
        self.logger = logging.getLogger('dev')

        if batch_size == 0:
            batch_size = 1  # later figure out what to do

        self.output_dataset_dir = py.join(output_dir, dataset)
        py.mkdir(self.output_dataset_dir)
        py.args_to_yaml(py.join(self.output_dataset_dir, 'settings.yml'), Namespace(
            epochs=epochs,
            epoch_decay=epoch_decay,
            pool_size=pool_size,
            output_dir=output_dir,
            datasets_dir=datasets_dir,
            dataset=dataset,
            image_ext=image_ext,
            crop_size=crop_size,
            load_size=load_size,
            batch_size=batch_size,
            adversarial_loss_mode=adversarial_loss_mode,  # ['gan', 'hinge_v1', 'hinge_v2', 'lsgan', 'wgan']
            lr=lr,
            gradient_penalty_mode=gradient_penalty_mode,  # ['none', 'dragan', 'wgan-gp'])
            gradient_penalty_weight=gradient_penalty_weight,
            cycle_loss_weight=cycle_loss_weight,
            identity_loss_weight=identity_loss_weight,
            beta_1=beta_1,
            color_depth=color_depth,
            progressive=progrssive)
                        )
        self.sample_dir = py.join(self.output_dataset_dir, 'samples_training')
        py.mkdir(self.sample_dir)

        self.epochs = epochs
        self.epoch_decay = epoch_decay
        self.pool_size = pool_size
        self.gradient_penalty_mode = gradient_penalty_mode
        self.gradient_penalty_weight = gradient_penalty_weight
        self.cycle_loss_weight = cycle_loss_weight
        self.identity_loss_weight = identity_loss_weight
        self.color_depth = color_depth
        self.adversarial_loss_mode = adversarial_loss_mode
        self.batch_size = batch_size
        self.beta_1 = beta_1
        self.color_depth = color_depth
        self.dataset = dataset
        self.datasets_dir = datasets_dir
        self.image_ext = image_ext
        self.progrssive = progrssive
        self.lr = lr

        self.A_img_paths = py.glob(py.join(datasets_dir, dataset, 'trainA'), '*.{}'.format(image_ext))
        self.B_img_paths = py.glob(py.join(datasets_dir, dataset, 'trainB'), '*.{}'.format(image_ext))

        # summary
        self.train_summary_writer = tf.summary.create_file_writer(py.join(self.output_dataset_dir, 'summaries', 'train'))
        # save settings

    def set_checkpoints(self):
        self.ep_cnt = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64)
        # checkpoint
        self.checkpoint = tl.Checkpoint(dict(G_A2B=self.G_A2B,
                                             G_B2A=self.G_B2A,
                                             D_A=self.D_A,
                                             D_B=self.D_B,
                                             G_optimizer=self.G_optimizer,
                                             D_optimizer=self.D_optimizer,
                                             ep_cnt=self.ep_cnt),
                                        py.join(self.output_dataset_dir, 'checkpoints'),
                                        max_to_keep=5)
        try:  # restore checkpoint including the epoch counter
            self.checkpoint.restore().assert_existing_objects_matched()
        except Exception as e:
            self.logger.warn(e)

    def construct_model(self, crop_size, load_size):
        self.A_B_dataset, len_dataset = data.make_zip_dataset(self.A_img_paths, self.B_img_paths, self.batch_size, load_size,
                                                              crop_size, training=True, repeat=False,
                                                              is_gray_scale=(self.color_depth == 1))
        self.len_dataset = len_dataset

        self.A2B_pool = data.ItemPool(self.pool_size)
        self.B2A_pool = data.ItemPool(self.pool_size)
        A_img_paths_test = py.glob(py.join(self.datasets_dir, self.dataset, 'testA'), '*.{}'.format(self.image_ext))
        B_img_paths_test = py.glob(py.join(self.datasets_dir, self.dataset, 'testB'), '*.{}'.format(self.image_ext))
        A_B_dataset_test, _ = data.make_zip_dataset(A_img_paths_test, B_img_paths_test, self.batch_size, load_size,
                                                    crop_size, training=False, repeat=True,
                                                    is_gray_scale=(self.color_depth == 1))
        self.test_iter = iter(A_B_dataset_test)
        self.G_A2B = module.ResnetGenerator(input_shape=(crop_size, crop_size, self.color_depth),
                                            output_channels=self.color_depth)
        self.G_B2A = module.ResnetGenerator(input_shape=(crop_size, crop_size, self.color_depth),
                                            output_channels=self.color_depth)
        self.D_A = module.ConvDiscriminator(input_shape=(crop_size, crop_size, self.color_depth))
        self.D_B = module.ConvDiscriminator(input_shape=(crop_size, crop_size, self.color_depth))
        self.d_loss_fn, self.g_loss_fn = gan.get_adversarial_losses_fn(self.adversarial_loss_mode)
        self.cycle_loss_fn = tf.losses.MeanAbsoluteError()
        self.identity_loss_fn = tf.losses.MeanAbsoluteError()
        self.G_lr_scheduler = module.LinearDecay(self.lr, self.epochs * self.len_dataset, self.epoch_decay * self.len_dataset)
        self.D_lr_scheduler = module.LinearDecay(self.lr, self.epochs * self.len_dataset, self.epoch_decay * self.len_dataset)
        self.G_optimizer = keras.optimizers.Adam(learning_rate=self.G_lr_scheduler, beta_1=self.beta_1)
        self.D_optimizer = keras.optimizers.Adam(learning_rate=self.D_lr_scheduler, beta_1=self.beta_1)

    def run(self, debug=False):
        # main loop
        with self.train_summary_writer.as_default():
            if self.progrssive:
                load_size = crop_size = 16
            self.construct_model(crop_size, load_size)
            # epoch counter
            self.set_checkpoints()
            self.train(debug)

    def train(self, debug):
        image_buffers = []
        image_buffers_test = []
        for ep in tqdm.trange(self.epochs, desc='Epoch Loop'):
            if ep < self.ep_cnt:
                continue

            # update epoch counter
            self.ep_cnt.assign_add(1)

            # train for an epoch
            for A, B in tqdm.tqdm(self.A_B_dataset, desc='Inner Epoch Loop', total=self.len_dataset):
                G_loss_dict, D_loss_dict = self.train_step(A, B)

                # # summary
                tl.summary(G_loss_dict, step=self.G_optimizer.iterations, name='G_losses')
                tl.summary(D_loss_dict, step=self.G_optimizer.iterations, name='D_losses')
                tl.summary({'learning rate': self.G_lr_scheduler.current_learning_rate},
                           step=self.G_optimizer.iterations,
                           name='learning rate')

                # sample
                snapshot_period = min(10, (self.epochs // 20) + 1)
                if self.G_optimizer.iterations.numpy() % snapshot_period == 0:
                    image_buffers.append(self.snapshot(A, B, 'train_iter-%09d.jpg', debug=debug))
                    A_test, B_test = next(self.test_iter)
                    image_buffers_test.append(self.snapshot(A_test, B_test, 'test_iter-%09d.jpg', debug=debug))

            # save checkpoint
            self.checkpoint.save(ep)
        if image_buffers:
            image_buffers.extend(image_buffers_test)
            make_animation(image_buffers, 'animations/cycleGAN')

    @tf.function
    def train_G(self, A, B):
        with tf.GradientTape() as t:
            A2B = self.G_A2B(A, training=True)
            B2A = self.G_B2A(B, training=True)
            A2B2A = self.G_B2A(A2B, training=True)
            B2A2B = self.G_A2B(B2A, training=True)
            A2A = self.G_B2A(A, training=True)
            B2B = self.G_A2B(B, training=True)

            A2B_d_logits = self.D_B(A2B, training=True)
            B2A_d_logits = self.D_A(B2A, training=True)

            A2B_g_loss = self.g_loss_fn(A2B_d_logits)
            B2A_g_loss = self.g_loss_fn(B2A_d_logits)
            A2B2A_cycle_loss = self.cycle_loss_fn(A, A2B2A)
            B2A2B_cycle_loss = self.cycle_loss_fn(B, B2A2B)
            A2A_id_loss = self.identity_loss_fn(A, A2A)
            B2B_id_loss = self.identity_loss_fn(B, B2B)

            G_loss = (A2B_g_loss + B2A_g_loss) + (A2B2A_cycle_loss + B2A2B_cycle_loss) * self.cycle_loss_weight + (
                    A2A_id_loss + B2B_id_loss) * self.identity_loss_weight

        G_grad = t.gradient(G_loss, self.G_A2B.trainable_variables + self.G_B2A.trainable_variables)
        self.G_optimizer.apply_gradients(zip(G_grad, self.G_A2B.trainable_variables + self.G_B2A.trainable_variables))

        return A2B, B2A, {'A2B_g_loss': A2B_g_loss,
                          'B2A_g_loss': B2A_g_loss,
                          'A2B2A_cycle_loss': A2B2A_cycle_loss,
                          'B2A2B_cycle_loss': B2A2B_cycle_loss,
                          'A2A_id_loss': A2A_id_loss,
                          'B2B_id_loss': B2B_id_loss}

    @tf.function
    def train_D(self, A, B, A2B, B2A):
        with tf.GradientTape() as t:
            A_d_logits = self.D_A(A, training=True)
            B2A_d_logits = self.D_A(B2A, training=True)
            B_d_logits = self.D_B(B, training=True)
            A2B_d_logits = self.D_B(A2B, training=True)

            A_d_loss, B2A_d_loss = self.d_loss_fn(A_d_logits, B2A_d_logits)
            B_d_loss, A2B_d_loss = self.d_loss_fn(B_d_logits, A2B_d_logits)
            D_A_gp = gan.gradient_penalty(functools.partial(self.D_A, training=True), A, B2A,
                                          mode=self.gradient_penalty_mode)
            D_B_gp = gan.gradient_penalty(functools.partial(self.D_B, training=True), B, A2B,
                                          mode=self.gradient_penalty_mode)

            D_loss = (A_d_loss + B2A_d_loss) + (B_d_loss + A2B_d_loss) + (
                    D_A_gp + D_B_gp) * self.gradient_penalty_weight

        D_grad = t.gradient(D_loss, self.D_A.trainable_variables + self.D_B.trainable_variables)
        self.D_optimizer.apply_gradients(zip(D_grad, self.D_A.trainable_variables + self.D_B.trainable_variables))

        return {'A_d_loss': A_d_loss + B2A_d_loss,
                'B_d_loss': B_d_loss + A2B_d_loss,
                'D_A_gp': D_A_gp,
                'D_B_gp': D_B_gp}

    def train_step(self, A, B):
        A2B, B2A, G_loss_dict = self.train_G(A, B)

        # cannot autograph `A2B_pool`
        A2B = self.A2B_pool(A2B)  # or A2B = A2B_pool(A2B.numpy()), but it is much slower
        B2A = self.B2A_pool(B2A)  # because of the communication between CPU and GPU

        D_loss_dict = self.train_D(A, B, A2B, B2A)

        return G_loss_dict, D_loss_dict

    @tf.function
    def sample(self, A, B):
        A2B = self.G_A2B(A, training=False)
        B2A = self.G_B2A(B, training=False)
        A2B2A = self.G_B2A(A2B, training=False)
        B2A2B = self.G_A2B(B2A, training=False)
        return A2B, B2A, A2B2A, B2A2B

    def snapshot(self, A, B, image_file_name, debug=False):
        A2B, B2A, A2B2A, B2A2B = self.sample(A, B)
        img = im.immerge(np.concatenate([A, A2B, A2B2A, B, B2A, B2A2B], axis=0), n_rows=2)
        im.imwrite(img, py.join(self.sample_dir, image_file_name % self.G_optimizer.iterations.numpy()))
        buffer = io.BytesIO()
        if self.color_depth == 1:
            pyplot.imshow(img.reshape(img.shape[0], img.shape[1]), cmap='gray')
        else:
            pyplot.imshow(img)
        pyplot.savefig(buffer, format='png')
        if debug:
            buffer.seek(0)
            pyplot.imread(buffer)
            pyplot.show()
        return buffer
