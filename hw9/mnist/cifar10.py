from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


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

  tf.summary.scalar('dropout_keep_probability', keep_prob)
  dropout = tf.layers.dropout(inputs=dense, rate=keep_prob)

  return tf.layers.dense(inputs=dropout, units=10)

def train(model_name="tutorial"):
  model_builder = {"tutorial":tutorial_model, "simple":simple_model}
  mnist = input_data.read_data_sets(FLAGS.data_dir)
  sess = tf.InteractiveSession()

  inputs_ = tf.placeholder(tf.float32, [None, 784], name='x-input')
  targets_ = tf.placeholder(tf.int64, [None], name='y-input')
  keep_prob = tf.placeholder(tf.float32)

  logits = model_builder[model_name](inputs_, keep_prob)
  
  cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=targets_, 
                    logits=logits))
  tf.summary.scalar('cross_entropy', cross_entropy)

  train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(cross_entropy)
  
  correct_prediction = tf.equal(tf.argmax(logits, 1), targets_)
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  tf.summary.scalar('accuracy', accuracy)

  # Merge all the summaries and write them out to
  # /tmp/tensorflow/mnist/logs/mnist_with_summaries (by default)
  merged = tf.summary.merge_all()
  base_log_dir = FLAGS.log_dir + '/' + model_name
  if tf.gfile.Exists(base_log_dir):
    tf.gfile.DeleteRecursively(base_log_dir)

  train_writer = tf.summary.FileWriter(base_log_dir + '/train', sess.graph)
  test_writer = tf.summary.FileWriter(base_log_dir + '/test')
  tf.global_variables_initializer().run()

  for i in range(FLAGS.max_steps):      
    train_data, train_labels = mnist.train.next_batch(100, shuffle = True)

    if i % 10 == 9:  # Record summaries and test-set accuracy
      run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
      run_metadata = tf.RunMetadata()
      summary, _ = sess.run([merged, train_step],
                              feed_dict={inputs_:train_data, targets_:train_labels, keep_prob:FLAGS.dropout},
                              options=run_options,
                              run_metadata=run_metadata)
      train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
      train_writer.add_summary(summary, i)

      summary, acc, loss = sess.run([merged, accuracy, cross_entropy], 
                                    feed_dict={inputs_:mnist.test.images, targets_:mnist.test.labels, keep_prob:1})
      test_writer.add_summary(summary, i)
      print('Accuracy at step %s: %s, Loss %s' % (i, acc, loss))
    else:
      summary, _ = sess.run([merged, train_step], 
                            feed_dict={inputs_:train_data, targets_:train_labels, keep_prob:FLAGS.dropout},)
      train_writer.add_summary(summary, i)

  train_writer.close()
  test_writer.close()


def main(unused_argv):
  #if tf.gfile.Exists(FLAGS.log_dir):
  #  tf.gfile.DeleteRecursively(FLAGS.log_dir)
  #tf.gfile.MakeDirs(FLAGS.log_dir)

  train(FLAGS.model)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--max_steps', type=int, default=2000,
                      help='Number of steps to run trainer.')
  parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='Initial learning rate')
  parser.add_argument('--dropout', type=float, default=0.4,
                      help='Keep probability for training dropout.')
  parser.add_argument('--model', type=str, default="tutorial",
                      help='Select model[tutorial, simple]')
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