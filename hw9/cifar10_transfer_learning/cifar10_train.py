# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A binary to train CIFAR-10 using a single GPU.

Accuracy:
cifar10_train.py achieves ~86% accuracy after 100K steps (256 epochs of
data) as judged by cifar10_eval.py.

Speed: With batch_size 128.

System        | Step Time (sec/batch)  |     Accuracy
------------------------------------------------------------------
1 Tesla K20m  | 0.35-0.60              | ~86% at 60K steps  (5 hours)
1 Tesla K40m  | 0.25-0.35              | ~86% at 100K steps (4 hours)

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from datetime import datetime
import time

import numpy as np
import tensorflow as tf

import cifar10
import cifar10_input

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp/cifar10_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('log_dir', os.path.join(os.getenv('TEST_TMPDIR', '/tmp'), 'tensorflow/cifar10/logs/enhance'),
                            """Summaries log directory.""")
tf.app.flags.DEFINE_integer('max_steps', 4000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 100,
                            """How often to log results to the console.""")


EVALUATE_BATCH_SIZE = 1000
def evaluate(session, correct_prediction):
  total_count = 0
  total_correct_count = 0

  for i in range(cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL // EVALUATE_BATCH_SIZE):
    correct_count = np.sum(session.run([correct_prediction]))
    total_correct_count += correct_count
    total_count += EVALUATE_BATCH_SIZE
  
  accuracy = total_correct_count / total_count

  return accuracy


def train():
  """Train CIFAR-10 for a number of steps."""
  with tf.Graph().as_default():
    global_step = tf.train.get_or_create_global_step()
    summary_writer = tf.summary.FileWriter(FLAGS.log_dir)

    # Get images and labels for CIFAR-10.
    # Force input pipeline to CPU:0 to avoid operations sometimes ending up on
    # GPU and resulting in a slow down.
    with tf.device('/cpu:0'):
      train_images, train_labels = cifar10.distorted_inputs()
      test_images, test_labels = cifar10_input.inputs(eval_data=True,
                                                data_dir=os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin'),
                                                batch_size=cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL)
      #test_images, test_labels = cifar10.inputs(True)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = cifar10.inference(train_images)

    prediction = cifar10.inference(test_images)
    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.cast(test_labels, tf.int64))
    test_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', test_accuracy)
    summary_op = tf.summary.merge_all()

    # Calculate loss.
    loss = cifar10.loss(logits, train_labels)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = cifar10.train(loss, global_step)

    class _LoggerHook(tf.train.SessionRunHook):
      """Logs loss and runtime."""

      def begin(self):
        self._step = -1
        self._start_time = time.time()

      def before_run(self, run_context):
        self._step += 1
        return tf.train.SessionRunArgs(loss)  # Asks for loss value.

      def after_run(self, run_context, run_values):
        if self._step % FLAGS.log_frequency == 0:
          current_time = time.time()
          duration = current_time - self._start_time
          self._start_time = current_time

          loss_value = run_values.results
          examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
          sec_per_batch = float(duration / FLAGS.log_frequency)

          format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                        'sec/batch)')
          print (format_str % (datetime.now(), self._step, loss_value,
                               examples_per_sec, sec_per_batch))

    with tf.train.MonitoredTrainingSession(
        hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
               tf.train.NanTensorHook(loss)],
        config=tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement)) as sess:
    #with tf.Session() as sess:
    #  for step in range(FLAGS.max_steps):
      while not sess.should_stop():
        step, _ = sess.run([global_step, train_op])
        if step % FLAGS.log_frequency == 0:
          test_acc, summary = sess.run([test_accuracy, summary_op])
          #test_acc = evaluate(sess, test_correct_prediction)
          print("Step:%4d Test Accuracy:%f"%(step, test_acc))
          summary_writer.add_summary(summary, step)

    summary_writer.close()

def main(argv=None):  # pylint: disable=unused-argument
  train()

if __name__ == '__main__':
  tf.app.run()
