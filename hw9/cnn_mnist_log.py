from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

def train(mnist):
  sess = tf.InteractiveSession()

  inputs_ = tf.placeholder(tf.float32, [None, 784], name='x-input')
  targets_ = tf.placeholder(tf.int64, [None], name='y-input')
  keep_prob = tf.placeholder(tf.float32)
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # MNIST images are 28x28 pixels, and have one color channel
  input_layer = tf.reshape(inputs_, [-1, 28, 28, 1])

  # Convolutional Layer #1
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 28, 28, 1]
  # Output Tensor Shape: [batch_size, 28, 28, 32]
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 28, 28, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 32]
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2
  # Computes 64 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 14, 14, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 64]
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #2
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 14, 14, 64]
  # Output Tensor Shape: [batch_size, 7, 7, 64]
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 7, 7, 64]
  # Output Tensor Shape: [batch_size, 7 * 7 * 64]
  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

  # Dense Layer
  # Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, 7 * 7 * 64]
  # Output Tensor Shape: [batch_size, 1024]
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

  # Add dropout operation; 0.6 probability that element will be kept
  tf.summary.scalar('dropout_keep_probability', keep_prob)
  dropout = tf.layers.dropout(inputs=dense, rate=keep_prob)

  # Logits layer
  # Input Tensor Shape: [batch_size, 1024]
  # Output Tensor Shape: [batch_size, 10]
  logits = tf.layers.dense(inputs=dropout, units=10) 

  cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=targets_, 
                    logits=logits)
  tf.summary.scalar('cross_entropy', cross_entropy)

  train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(cross_entropy)
  
  #accuracy = tf.metrics.accuracy(labels=targets_, predictions=tf.argmax(input=logits, axis=1))
  #with tf.name_scope('accuracy'):
  #  with tf.name_scope('correct_prediction'):
  #    correct_prediction = tf.equal(tf.argmax(logits, 1), targets_)
  #  with tf.name_scope('accuracy'):
  #    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  correct_prediction = tf.equal(tf.argmax(logits, 1), targets_)
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  tf.summary.scalar('accuracy', accuracy)

  # Merge all the summaries and write them out to
  # /tmp/tensorflow/mnist/logs/mnist_with_summaries (by default)
  merged = tf.summary.merge_all()
  train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
  test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')
  tf.global_variables_initializer().run()

  def feed_dict(train):
    """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
    if train:
      xs, ys = mnist.train.next_batch(100)
      k = FLAGS.dropout
    else:
      xs, ys = mnist.test.images, mnist.test.labels
      k = 1.0
    return {inputs_: xs, targets_: ys, keep_prob: k}
  
  for i in range(FLAGS.max_steps):      
    if i % 100 == 99:  # Record summaries and test-set accuracy
      run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
      run_metadata = tf.RunMetadata()
      summary, _ = sess.run([merged, train_step],
                              feed_dict=feed_dict(True),
                              options=run_options,
                              run_metadata=run_metadata)
      train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
      train_writer.add_summary(summary, i)

      summary, acc, loss = sess.run([merged, accuracy, cross_entropy], feed_dict=feed_dict(False))
      test_writer.add_summary(summary, i)
      print('Accuracy at step %s: %s, Loss %s' % (i, acc, loss))
    else:
      summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
      train_writer.add_summary(summary, i)

  train_writer.close()
  test_writer.close()


def main(unused_argv):
  if tf.gfile.Exists(FLAGS.log_dir):
    tf.gfile.DeleteRecursively(FLAGS.log_dir)
  tf.gfile.MakeDirs(FLAGS.log_dir)

  mnist = input_data.read_data_sets(FLAGS.data_dir)
  train(mnist)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--max_steps', type=int, default=2000,
                      help='Number of steps to run trainer.')
  parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='Initial learning rate')
  parser.add_argument('--dropout', type=float, default=0.4,
                      help='Keep probability for training dropout.')
  parser.add_argument(
      '--data_dir',
      type=str,
      default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                           'tensorflow/mnist/input_data'),
      help='Directory for storing input data')
  parser.add_argument(
      '--log_dir',
      type=str,
      default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                           'tensorflow/mnist/logs/mnist'),
      help='Summaries log directory')
  FLAGS, unparsed = parser.parse_known_args()

  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)