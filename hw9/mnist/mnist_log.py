import os
import time
from datetime import datetime

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('max_steps', 2000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('batch_size', 100,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('log_frequency', 10,
                            """How often to log results to the console.""")
tf.app.flags.DEFINE_float('dropout', 0.4,
                            """Keep probability for training dropout.""")
tf.app.flags.DEFINE_float('learning_rate', 0.001,
                            """Initial learning rate.""")
tf.app.flags.DEFINE_string('log_dir', os.path.join(os.getenv('TEST_TMPDIR', '/tmp'), 'tensorflow/mnist/logs/mnist'),
                            """Summaries log directory.""")

def inference(images, labels):
  # Input Layer
  input_layer = tf.reshape(images, [-1, 28, 28, 1])

  # Convolutional Layer #1
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #2
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Flatten tensor into a batch of vectors
  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

  # Dense Layer
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

  # Add dropout operation; 0.6 probability that element will be kept
  dropout = tf.layers.dropout(inputs=dense, rate=0.4)

  # Logits layer
  return tf.layers.dense(inputs=dropout, units=10)

def loss(logits, labels):
  # Calculate Loss
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits, name='cross_entropy_per_example')
  return  tf.reduce_mean(cross_entropy, name='cross_entropy')

def train(mnist):
  global_step = tf.train.get_or_create_global_step()
  inputs_ = tf.placeholder(tf.float32, [None, 784], name='image-inputs')
  targets_ = tf.placeholder(tf.int64, [None], name='image-labels')
  keep_prob = tf.placeholder(tf.float32)
  
  logits = inference(inputs_, keep_prob)
  cross_entropy = loss(logits, targets_)

  # Training Op
  #optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
  #train_op = optimizer.minimize(loss=loss_, global_step=global_step)
  train_op = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(cross_entropy, global_step=global_step)

  accuracy = tf.metrics.accuracy(labels=targets_, predictions=tf.argmax(input=logits, axis=1))
  tf.summary.scalar('accuracy', accuracy)

  class _LoggerHook(tf.train.SessionRunHook):
      """Logs loss and runtime."""
      def begin(self):
        self._step = -1
        self._start_time = time.time()

      def before_run(self, run_context):
        self._step += 1
        return tf.train.SessionRunArgs(cross_entropy)  # Asks for loss value

      def after_run(self, run_context, run_values):
        if self._step % FLAGS.log_frequency == 0:
          current_time = time.time()
          duration = current_time - self._start_time
          self._start_time = current_time

          loss_value = run_values.results[0]
          examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
          sec_per_batch = float(duration / FLAGS.log_frequency)

          format_str = ('%s: step %d, loss = %f (%.1f examples/sec; %.3f '
                        'sec/batch)')
          print (format_str % (datetime.now(), self._step, loss_value,
                               examples_per_sec, sec_per_batch))  
                               
  class _SummaryHook(tf.train.SummarySaverHook):
    def begin(self):
      self._step = -1
      self._start_time = time.time()
    
    def before_run(self, run_context):
      self._step += 1

      #feed_dict = {inputs_:mnist.test.images, targets_:mnist.test.labels, keep_prob:1}

      return tf.train.SessionRunArgs(accuracy)  # Asks for accuracy value

    def after_run(self, run_context, run_values):
      accuracy_value = run_values.results[1]

      print ('step %d, accuracy = %f' % (self._step, accuracy_value))

  #log_writer = tf.summary.FileWriter(FLAGS.log_dir + '/' + model_name + '/train', sess.graph)
  summary_op = tf.summary.merge_all()
  summary_hook = _SummaryHook(output_dir=FLAGS.log_dir, 
                    summary_op=summary_op, save_steps=FLAGS.log_frequency)

  test_dict = {inputs_:mnist.test.images, targets_:mnist.test.labels, keep_prob:1}

  with tf.train.MonitoredSession(
        hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
                tf.train.NanTensorHook(cross_entropy)]) as mon_sess:
        while not mon_sess.should_stop():
            batch_data, batch_labels = mnist.train.next_batch(100, shuffle = True)
            step, _ = mon_sess.run([global_step, train_op], feed_dict = {inputs_:batch_data, targets_:batch_labels, 
                keep_prob:FLAGS.dropout})
            
            if(step % FLAGS.log_frequency == 0):
              accuracy_value = mon_sess.run(accuracy, feed_dict=test_dict)
              print(accuracy_value)
              #print('step %d, test accuracy = %f'%(step, accuracy_value))


def main(unused_argv):
  # Load training and eval data
  mnist = input_data.read_data_sets("/tmp/tensorflow/mnist/input_data")
  #mnist = tf.contrib.learn.datasets.load_dataset("mnist")

  train(mnist)

if __name__ == "__main__":
  tf.app.run()