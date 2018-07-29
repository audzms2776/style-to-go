import tensorflow as tf
from tensorflow import keras
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from torchvision.utils import save_image
import torch
from tqdm import tqdm

mnist = input_data.read_data_sets("/tmp/data", one_hot=True)

# tf.enable_eager_execution()

batch_size = 100
noise_size = 64
total_step = 10000


def bce_loss(labels, predicts):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=predicts))


def get_noise():
    return np.random.normal(0, 1., size=[batch_size, noise_size]).astype(np.float32)


def G(z, re):
    with tf.variable_scope("gggg", reuse=re):
        out = keras.layers.Dense(256, activation=tf.nn.relu)(z)
        # out = keras.layers.Dense(512, activation=tf.nn.relu)(out)
        out = keras.layers.Dense(784, activation=tf.nn.tanh)(out)

    return out


def D(x, re):
    with tf.variable_scope("dddd", reuse=re):
        out = keras.layers.Dense(256, activation=tf.nn.relu)(x)
        # out = keras.layers.Dense(64, activation=tf.nn.relu)(out)
        out = keras.layers.Dense(1)(out)

    return out


x_data = tf.placeholder(tf.float32, [None, 784])
z_noise = tf.placeholder(tf.float32, [None, noise_size])

# case 1
# d_loss = tf.reduce_mean(D(G(z_noise, False), False)) - tf.reduce_mean(D(x_data, True))
# g_loss = - tf.reduce_mean(D(G(z_noise, True), True))

# case 2
fake_img = G(z_noise, False)
d_fake = D(fake_img, False)
d_data = D(x_data, True)

d_loss = bce_loss(tf.ones_like(d_data), d_data) + bce_loss(tf.zeros_like(d_fake), d_fake)
g_loss = bce_loss(tf.zeros_like(d_fake), d_fake)

fake_img_list = tf.reshape(fake_img, (batch_size, 1, 28, 28))

optimizer = tf.train.AdamOptimizer(0.0002)

t_vars = tf.trainable_variables()

d_var_list = [var for var in t_vars if "dddd" in var.name]
g_var_list = [var for var in t_vars if "gggg" in var.name]

d_train = optimizer.minimize(d_loss, var_list=d_var_list)
g_train = optimizer.minimize(g_loss, var_list=g_var_list)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

train_pbar = tqdm(total=total_step)

for step in range(total_step):
    train_pbar.update(1)
    batch_xs, _ = mnist.train.next_batch(batch_size)
    sample_noise = get_noise()

    d_error, _ = sess.run([d_loss, d_train],
                          feed_dict={
                              x_data: batch_xs,
                              z_noise: sample_noise
                          })

    g_error, _ = sess.run([g_loss, g_train],
                          feed_dict={
                              z_noise: sample_noise
                          })

    if (step + 1) % 100 == 0:
        train_pbar.set_description('d error {} g error {}'.format(d_error, g_error))

test_noise = get_noise()
result_img = sess.run(fake_img_list, feed_dict={
    z_noise: test_noise
})

print(result_img.shape)
result_tensor = torch.from_numpy(result_img)
save_image(result_tensor, 'result.png', normalize=True)
