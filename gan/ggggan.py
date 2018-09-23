import tensorflow as tf
import numpy as np
import torch
import torchvision.utils as vutils
from tqdm import tqdm
from tensorflow.examples.tutorials.mnist import input_data
from tensorboardX import SummaryWriter

mnist = input_data.read_data_sets("/tmp/", one_hot=True)

noise_size = 100
total_epoch = 200


def get_noise(batch_size):
    return np.random.normal(0, 1., size=[batch_size, noise_size]).astype(np.float32)


def gan_log_loss(pos, neg, name='gan_log_loss'):
    """
    log loss function for GANs.
    - Generative Adversarial Networks: https://arxiv.org/abs/1406.2661
    """
    with tf.variable_scope(name):
        # generative model G
        gl = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=neg, labels=tf.ones_like(neg)))
        # discriminative model D
        d_loss_pos = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=pos, labels=tf.ones_like(pos)))
        d_loss_neg = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=neg, labels=tf.zeros_like(neg)))
        pos_acc = tf.reduce_mean(tf.sigmoid(pos))
        neg_acc = tf.reduce_mean(tf.sigmoid(neg))
        # loss
        dl = tf.add(.5 * d_loss_pos, .5 * d_loss_neg)
    return gl, dl, pos_acc, neg_acc


def dense_layer(in_tensor, out_size, name, activation=True, batch_norm=True):
    x = tf.layers.dense(in_tensor, out_size, name=name)

    if activation:
        x = tf.nn.leaky_relu(x)

    if batch_norm:
        x = tf.layers.batch_normalization(x, name=name)

    return x


def gen(noise, reuse=False):
    with tf.variable_scope('gen') as scope:
        if reuse:
            scope.reuse_variables()

        layer1 = dense_layer(noise, 256, name='g_layer1')
        layer2 = dense_layer(layer1, 512, name='g_layer2')
        layer3 = dense_layer(layer2, 1024, name='g_layer3')

        result_img = tf.nn.tanh(dense_layer(layer3, 784, activation=False,
                                            batch_norm=False, name='g_layer4'))

    return result_img


def dis(img, reuse=False):
    with tf.variable_scope('gen') as scope:
        if reuse:
            scope.reuse_variables()

        layer1 = tf.nn.leaky_relu(tf.layers.dense(img, 512, name='d_layer1'))
        layer2 = tf.nn.leaky_relu(tf.layers.dense(layer1, 256, name='d_layer2'))
        layer3 = tf.nn.leaky_relu(tf.layers.dense(layer2, 64), name='d_layer3')
        result = tf.layers.dense(layer3, 1, name='d_layer4')

    return result


noise_input = tf.placeholder(tf.float32, [None, noise_size])
real_input = tf.placeholder(tf.float32, [None, 784])

# real img
D_real = dis(real_input, reuse=False)

# fake img
G_img = gen(noise_input, reuse=False)
D_fake = dis(G_img, reuse=True)

g_loss, d_loss, t_acc, f_acc = gan_log_loss(D_real, D_fake)

t_vars = tf.trainable_variables()
d_vars = [var for var in t_vars if 'd_' in var.name]
g_vars = [var for var in t_vars if 'g_' in var.name]

d_optimization = tf.train.AdamOptimizer(learning_rate=0.0002).minimize(d_loss, var_list=d_vars)
g_optimization = tf.train.AdamOptimizer(learning_rate=0.0002).minimize(g_loss, var_list=g_vars)

sample_img = gen(noise_input, reuse=True)
saver = tf.train.Saver()

writer = SummaryWriter()
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in tqdm(range(total_epoch)):
    for idx in range(60):
        batch_xs, _ = mnist.train.next_batch(100)
        batch_z = get_noise(100)

        # update D
        sess.run(d_optimization, feed_dict={noise_input: batch_z,
                                            real_input: batch_xs})

        # update G
        sess.run(g_optimization, feed_dict={noise_input: batch_z})
        # sess.run(g_optimization, feed_dict={noise_input: batch_z})

        # calc loss
        dis_loss = sess.run(t_acc, feed_dict={real_input: batch_xs,
                                               noise_input: batch_z})
        gen_loss = sess.run(f_acc, feed_dict={noise_input: batch_z})

    writer.add_scalar('data/d_loss', dis_loss, epoch)
    writer.add_scalar('data/g_loss', gen_loss, epoch)

    # sample
    if (epoch + 1) % 10 == 0:
        sample = sess.run(sample_img, feed_dict={noise_input: batch_z})
        re_sample = torch.from_numpy(np.reshape(sample, (100, 1, 28, 28)))
        x = vutils.make_grid(re_sample, normalize=True, scale_each=True)
        writer.add_image('Image{}'.format(epoch), x, epoch)

writer.close()
