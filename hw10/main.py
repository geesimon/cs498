import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import vae
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', validation_size=0)

dim_z = 20
n_hidden = 500
learn_rate = 1e-3
dim_img = mnist.train.images.shape[1]

x = tf.placeholder(tf.float32, shape=[None, dim_img], name='target_img')

# dropout
keep_prob = tf.placeholder(tf.float32, name='keep_prob')

# input for PMLR
z_in = tf.placeholder(tf.float32, shape=[None, dim_z], name='latent_variable')

# network architecture
y, z, loss, neg_marginal_likelihood, KL_divergence = vae.autoencoder(x, x, dim_img, dim_z, n_hidden, keep_prob)

# optimization
train_op = tf.train.AdamOptimizer(learn_rate).minimize(loss)

decoded = vae.decoder(z_in, dim_img, n_hidden)


batch_size = 128
num_epochs = 20
batch_size = 128
total_batch = mnist.train.images.shape[0]//batch_size


#Prepare digit images for testing
SameDigitPairs = mnist.train.images[np.argwhere(mnist.train.labels == 2)[100:120]].reshape(-1, dim_img)
DiffDigitParis = np.zeros((20, dim_img))
for i in range(10):
    images = mnist.train.images[np.argwhere(mnist.train.labels == i)[200:202]].reshape(-1, dim_img)

    DiffDigitParis[i * 2, :] = images[0]
    DiffDigitParis[i * 2 + 1, :] = images[1]

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer(), feed_dict={keep_prob : 0.9})
    for epoch in range(num_epochs):
        for i in range(total_batch):
            train_images, train_labels = mnist.train.next_batch(batch_size)
            _, tot_loss, loss_likelihood, loss_divergence = sess.run(
                    (train_op, loss, neg_marginal_likelihood, KL_divergence),
                    feed_dict={x: train_images, keep_prob : 0.9})
            
        print("epoch %d: L_tot %03.2f L_likelihood %03.2f L_divergence %03.2f" % (epoch, tot_loss, loss_likelihood, loss_divergence))
    
    
    same_z_codes = sess.run((z), feed_dict={x: SameDigitPairs, keep_prob : 1})
    diff_z_codes = sess.run((z), feed_dict={x: DiffDigitParis, keep_prob : 1})
