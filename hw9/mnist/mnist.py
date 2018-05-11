from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('max_steps', 2000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('batch_size', 100,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('log_frequency', 100,
                            """How often to log results to the console.""")
tf.app.flags.DEFINE_float('dropout', 0.4,
                            """Keep probability for training dropout.""")
tf.app.flags.DEFINE_float('learning_rate', 0.001,
                            """Initial learning rate.""")
tf.app.flags.DEFINE_string('log_dir', os.path.join(os.getenv('TEST_TMPDIR', '/tmp'), 'tensorflow/mnist/logs/'),
                            """Summaries log directory.""")
tf.app.flags.DEFINE_string('data_dir', os.path.join(os.getenv('TEST_TMPDIR', '/tmp'), 'tensorflow/mnist/input_data'),
                            """MNIST data directory.""")
tf.app.flags.DEFINE_string('model_dir', os.path.join(os.getenv('TEST_TMPDIR', '/tmp'), 'tensorflow/mnist/model/'),
                            """Model saving directory.""")
tf.app.flags.DEFINE_string('model_name', "tutorial",
                            """Specify model name for training.""")

def simple_model(inputs, keep_prob):
  hidden1 = tf.layers.dense(inputs=inputs, units=500, activation=tf.nn.relu)
  dropped = tf.nn.dropout(hidden1, keep_prob)
  return tf.layers.dense(dropped, units=10, activation=tf.identity)

def tutorial_model(inputs, keep_prob):  
  input_layer = tf.reshape(inputs, [-1, 28, 28, 1])
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
  
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
  dropout = tf.layers.dropout(inputs=dense, rate=keep_prob)

  return tf.layers.dense(inputs=dropout, units=10)

def enhance_model(inputs, keep_prob):
  input_layer = tf.reshape(inputs, [-1, 28, 28, 1])
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=8,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)
  norm1 = tf.layers.batch_normalization(conv1)
  pool1 = tf.layers.max_pooling2d(inputs=norm1, pool_size=[2, 2], strides=2)

  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=32,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)
  norm2 = tf.layers.batch_normalization(conv2)
  pool2 = tf.layers.max_pooling2d(inputs=norm2, pool_size=[2, 2], strides=2)

  conv3 = tf.layers.conv2d(
      inputs=pool2,
      filters=64,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)
  norm3 = tf.layers.batch_normalization(conv3)
  pool3 = tf.layers.max_pooling2d(inputs=norm3, pool_size=[2, 2], strides=2)
  pool3_flat = tf.reshape(pool3, [-1, 3 * 3 * 64])

  dense = tf.layers.dense(inputs=pool3_flat, units=1024, activation=tf.nn.relu)
  dropout = tf.layers.dropout(inputs=dense, rate=keep_prob)

  return tf.layers.dense(inputs=dropout, units=10)

def train(model_name):
  log_dir = FLAGS.log_dir + '/' + model_name
  if tf.gfile.Exists(log_dir):
    tf.gfile.DeleteRecursively(log_dir)

  model_builder = {"tutorial":tutorial_model, "adam":tutorial_model,
                  "enhance":enhance_model}
  optimizer_builder = {"tutorial":tf.train.GradientDescentOptimizer, 
                      "adam": tf.train.AdamOptimizer,
                      "enhance": tf.train.AdamOptimizer}

  mnist = input_data.read_data_sets(FLAGS.data_dir)
  sess = tf.InteractiveSession()

  inputs_ = tf.placeholder(tf.float32, [None, 784], name='x-input')
  targets_ = tf.placeholder(tf.int64, [None], name='y-input')
  keep_prob = tf.placeholder(tf.float32)

  logits = model_builder[model_name](inputs_, keep_prob)
  
  cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets_, 
                    logits=logits))
  tf.summary.scalar('cross_entropy', cross_entropy)

  train_step = optimizer_builder[model_name](FLAGS.learning_rate).minimize(cross_entropy)
  
  correct_prediction = tf.equal(tf.argmax(logits, 1), targets_)
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  
  tf.summary.scalar('accuracy', accuracy)

  # Merge all the summaries and write them out to
  merged = tf.summary.merge_all()
  summary_writer = tf.summary.FileWriter(log_dir + '/test')
  tf.global_variables_initializer().run()

  for i in range(FLAGS.max_steps + 1):      
    train_data, train_labels = mnist.train.next_batch(FLAGS.batch_size, shuffle = True)
    train_dict = {inputs_:train_data, targets_:train_labels, keep_prob:FLAGS.dropout}
    test_dict = {inputs_:mnist.test.images, targets_:mnist.test.labels, keep_prob:1}

    if i % FLAGS.log_frequency == 0:  # Record summaries and test-set accuracy
      run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
      run_metadata = tf.RunMetadata()
      loss, _ = sess.run([cross_entropy, train_step], feed_dict=train_dict,
                        options=run_options, run_metadata=run_metadata)
      summary_writer.add_run_metadata(run_metadata, 'step%03d' % i)
      
      summary, test_acc = sess.run([merged, accuracy], feed_dict=test_dict)
      summary_writer.add_summary(summary, i)
      print('Test Accuracy at Step %4d: %f Loss %f' % (i, test_acc, loss))
    else:
      sess.run([train_step], feed_dict=train_dict)

  summary_writer.close()

def main(unused_argv):
  train(FLAGS.model_name)

if __name__ == "__main__":
  tf.app.run()