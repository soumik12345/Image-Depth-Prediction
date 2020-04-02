from time import time
from tqdm import tqdm
import tensorflow as tf
from src.losses import *
from src.utils import *
from src.dataset import *
from src.models import *
from os.path import join


class ImageDepthModel:

    def __init__(self, train_dataset, val_dataset, test_dataset, config_file='./config.json'):
        self.config = parse_config(config_file)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.generator = Generator()
        self.discriminator = Discriminator()
        self.generator_optimizer, self.discriminator_optimizer = self.get_optimizers()
        self.checkpoint, self.checkpoint_prefix = self.get_checkpoint(self.config['checkpoint_dir'])
        self.summary_writer = tf.summary.create_file_writer(logdir=self.config['log_dir'])

    def get_optimizers(self):
        generator_optimizer = tf.keras.optimizers.Adam(self.config['lr'], beta_1=0.5)
        discriminator_optimizer = tf.keras.optimizers.Adam(self.config['lr'], beta_1=0.5)
        return discriminator_optimizer, generator_optimizer

    def get_checkpoint(self, checkpoint_dir='./training_checkpoints'):
        checkpoint_prefix = join(checkpoint_dir, 'ckpt')
        checkpoint = tf.train.Checkpoint(
            generator_optimizer=self.generator_optimizer,
            discriminator_optimizer=self.discriminator_optimizer,
            generator=self.generator, discriminator=self.discriminator
        )
        return checkpoint, checkpoint_prefix

    @tf.function
    def train_step(self, input_image, target):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = self.generator(input_image, training=True)
            disc_real_output = self.discriminator([input_image, target], training=True)
            disc_generated_output = self.discriminator([input_image, gen_output], training=True)
            gen_loss = generator_loss(disc_generated_output, gen_output, target)
            disc_loss = discriminator_loss(disc_real_output, disc_generated_output)
        generator_gradients = gen_tape.gradient(
            gen_loss, self.generator.trainable_variables
        )
        discriminator_gradients = disc_tape.gradient(
            disc_loss, self.discriminator.trainable_variables
        )
        self.generator_optimizer.apply_gradients(
            zip(
                generator_gradients,
                self.generator.trainable_variables
            )
        )
        self.discriminator_optimizer.apply_gradients(
            zip(
                discriminator_gradients,
                self.discriminator.trainable_variables
            )
        )
        return gen_loss, disc_loss

    def train(self, epochs, checkpoint_step=5):
        with self.summary_writer.as_default():
            for epoch in range(1, epochs + 1):
                start = time()
                print('Epoch', str(epoch), 'going on....')
                iteration = 0
                for input_image, target in tqdm(self.train_dataset):
                    gen_loss, disc_loss = self.train_step(input_image, target)
                    iteration += 1
                    tf.summary.scalar('train/summary/generator_loss', gen_loss, iteration)
                    tf.summary.scalar('train/summary/discriminator_loss', disc_loss, iteration)
                print('Completed.')
                if (epoch + 1) % checkpoint_step == 0:
                    self.checkpoint.save(file_prefix=self.checkpoint_prefix)
                for example_input, example_target in self.val_dataset.take(1):
                    prediction = self.generator(example_input)
                    tf.summary.image('train/val_dataset/image', example_input * 0.5 + 0.5, epoch)
                    tf.summary.image('train/val_dataset/ground_truth', example_target * 0.5 + 0.5, epoch)
                    tf.summary.image('train/val_dataset/prediction', prediction * 0.5 + 0.5, epoch)
                print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, time() - start))

    def resize(self, input_image, real_image=None, height=256, width=256):
        input_image = tf.image.resize(
            input_image, [height, width],
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
        )
        if real_image is not None:
            real_image = tf.image.resize(
                real_image, [height, width],
                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
            )
            return input_image, real_image
        return input_image

    def normalize(self, input_image, real_image=None):
        input_image = (input_image / 127.5) - 1
        if real_image is not None:
            real_image = (real_image / 127.5) - 1
            return input_image, real_image
        return input_image

    def predict(self, image_file, paired_file=False):
        image = tf.io.read_file(image_file)
        image = tf.image.decode_jpeg(image)
        image.set_shape([None, None, 3])
        if paired_file:
            w = tf.shape(image)[1]
            w = w // 2
            real_image = image[:, w:, :]
            input_image = image[:, : w, :]
            input_image = tf.cast(input_image, tf.float32)
            real_image = tf.cast(real_image, tf.float32)
            input_image, real_image = self.resize(input_image, real_image)
            input_image, real_image = self.normalize(input_image, real_image)
            prediction = self.generator(tf.expand_dims(input_image, axis=0))
            return input_image * 0.5 + 0.5, real_image * 0.5 + 0.5, prediction * 0.5 + 0.5
        input_image = image
        input_image = self.resize(input_image)
        input_image = self.normalize(input_image)
        prediction = self.generator(tf.expand_dims(input_image, axis=0))
        return input_image * 0.5 + 0.5, prediction * 0.5 + 0.5
