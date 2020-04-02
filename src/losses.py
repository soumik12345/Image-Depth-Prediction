import tensorflow as tf


loss = tf.keras.losses.BinaryCrossentropy(from_logits = True)


def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = loss(tf.zeros_like(disc_generated_output), disc_generated_output)
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss


def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss(tf.ones_like(disc_generated_output), disc_generated_output)
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    total_gen_loss = gan_loss + (100 * l1_loss)
    return total_gen_loss
