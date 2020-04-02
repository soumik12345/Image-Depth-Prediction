import tensorflow as tf
from .utils import parse_config
from matplotlib import pyplot as plt


class DepthDataset:

    def __init__(self, config_file):
        self.config = parse_config(config_file)

    def load(self, image_file):
        image = tf.io.read_file(image_file)
        image = tf.image.decode_jpeg(image)
        image.set_shape([None, None, 3])
        w = tf.shape(image)[1]
        w = w // 2
        real_image = image[:, w:, :]
        input_image = image[:, : w, :]
        input_image = tf.cast(input_image, tf.float32)
        real_image = tf.cast(real_image, tf.float32)
        return input_image, real_image

    def resize(self, input_image, real_image, height, width):
        input_image = tf.image.resize(
            input_image, [height, width],
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
        )
        real_image = tf.image.resize(
            real_image, [height, width],
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
        )
        return input_image, real_image

    def random_crop(self, input_image, real_image):
        stacked_image = tf.stack([input_image, real_image], axis=0)
        cropped_image = tf.image.random_crop(
            stacked_image,
            size=[
                2, self.config['image_height'],
                self.config['image_width'], 3
            ]
        )
        return cropped_image[0], cropped_image[1]

    def normalize(self, input_image, real_image):
        input_image = (input_image / 127.5) - 1
        real_image = (real_image / 127.5) - 1
        return input_image, real_image

    @tf.function()
    def augmentation(self, input_image, real_image):
        input_image, real_image = self.resize(input_image, real_image, 286, 286)
        input_image, real_image = self.random_crop(input_image, real_image)
        if tf.random.uniform(()) > 0.5:
            input_image = tf.image.flip_left_right(input_image)
            real_image = tf.image.flip_left_right(real_image)
        return input_image, real_image

    def visualize(self, image_file, augment=False):
        input_image, real_image = self.load(image_file)
        if augment:
            input_image, real_image = self.augmentation(input_image, real_image)
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 16))
        plt.setp(axes.flat, xticks=[], yticks=[])
        for i, ax in enumerate(axes.flat):
            if i % 2 == 0:
                ax.imshow(input_image.numpy() / 255.0)
                ax.set_xlabel('Input_Image')
            else:
                ax.imshow(real_image.numpy() / 255.0)
                ax.set_xlabel('Real_Image')
        plt.show()

    def load_image_train(self, image_file):
        input_image, real_image = self.load(image_file)
        input_image, real_image = self.augmentation(input_image, real_image)
        input_image, real_image = self.normalize(input_image, real_image)
        return input_image, real_image

    def load_image_test(self, image_file):
        input_image, real_image = self.load(image_file)
        input_image, real_image = self.resize(
            input_image, real_image,
            self.config['image_height'],
            self.config['image_width']
        )
        input_image, real_image = self.normalize(input_image, real_image)
        return input_image, real_image

    def get_datasets(
            self, train_files, val_files, test_files,
            buffer_size=400, batch_size=8):
        train_dataset = tf.data.Dataset.list_files(train_files)
        train_dataset = train_dataset.map(
            self.load_image_train,
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        train_dataset = train_dataset.shuffle(buffer_size)
        train_dataset = train_dataset.batch(batch_size)
        val_dataset = tf.data.Dataset.list_files(val_files)
        val_dataset = val_dataset.map(self.load_image_test)
        val_dataset = val_dataset.batch(1)
        test_dataset = tf.data.Dataset.list_files(test_files)
        test_dataset = test_dataset.map(self.load_image_test)
        test_dataset = test_dataset.batch(1)
        return train_dataset, val_dataset, test_dataset
