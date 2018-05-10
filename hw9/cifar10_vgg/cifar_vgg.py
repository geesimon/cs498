import tensorflow as tf
import cifar10_input
import os

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_dir', '/tmp/cifar10_data/cifar-10-batches-bin',
                           """Path to the CIFAR-10 data directory.""")
tf.app.flags.DEFINE_string('log_dir', os.path.join(os.getenv('TEST_TMPDIR', '/tmp'), 'tensorflow/cifar10/logs/TL'),
                            """Summaries log directory.""")
tf.app.flags.DEFINE_integer('max_steps', 5000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 100,
                            """How often to log results to the console.""")
tf.app.flags.DEFINE_integer('batch_size', 100,
                            """Number of images to process in a batch.""")


NUM_CLASSES = cifar10_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL


# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.

def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var


def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """
  dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
  var = _variable_on_cpu(
      name,
      shape,
      tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var

def avg_pool_layer(bottom, name):
    return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

def max_pool_layer(bottom, name):
    return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

def norm_layer(bottom, name):
    return tf.nn.lrn(bottom, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)

def dropout_layer(bottom, keep_prob, name):
    return tf.nn.dropout(bottom, keep_prob, name=name)

def conv_layer(bottom, kernel_size, layer_num, name):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
        kernel = _variable_with_weight_decay('weights',
                                            shape=[kernel_size[0], kernel_size[1], bottom.shape[-1].value, layer_num],
                                            stddev=5e-2,
                                            wd=None)
        conv = tf.nn.conv2d(bottom, kernel, [1, kernel_size[0], kernel_size[1], 1], padding='SAME')
        biases = _variable_on_cpu('biases', [layer_num], tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(conv, biases)
        return tf.nn.relu(pre_activation, name=scope.name)

def fc_layer(bottom, unit_num, name):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
        shape = bottom.get_shape().as_list()
        dim = 1
        for d in shape[1:]:
            dim *= d
        reshape = tf.reshape(bottom, [dim, -1])
        weights = _variable_with_weight_decay('weights', shape=[dim, unit_num],
                                            stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', [unit_num], tf.constant_initializer(0.1))
        return tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)


def build_vgg_model(images):
    conv1_1     = conv_layer(images,  (3, 3), 64, name="conv1_1")
    norm1_1     = norm_layer(conv1_1, name="norm1_1")
    dropout1_1  = dropout_layer(norm1_1, 0.3, name="dropout1_1")
    conv1_2     = conv_layer(dropout1_1,  (3, 3), 64, "conv1_2")
    norm1_2     = norm_layer(conv1_2, name="norm1_2")
    pool1       = max_pool_layer(norm1_2, name="pool1")
    
    conv2_1     = conv_layer(pool1,  (3, 3), 128, name="conv2_1")
    norm2_1     = norm_layer(conv2_1, name="norm2_1")
    dropout2_1  = dropout_layer(norm2_1, 0.4, name="dropout2_1")
    conv2_2     = conv_layer(dropout2_1,  (3, 3), 128, name="conv2_2")
    norm2_2     = norm_layer(conv2_2, name="norm2_2")
    pool2       = max_pool_layer(norm2_2, name="pool2")

    conv3_1     = conv_layer(pool2,  (3, 3), 256, name="conv3_1")
    norm3_1     = norm_layer(conv3_1, name="norm3_1")
    dropout3_1  = dropout_layer(norm3_1, 0.4, name="dropout3_1")
    conv3_2     = conv_layer(dropout3_1,  (3, 3), 256, name="conv3_2")
    norm3_2     = norm_layer(conv3_2, name="norm3_2")
    dropout3_2  = dropout_layer(norm3_2, 0.4, name="dropout3_2")
    conv3_3     = conv_layer(dropout3_2,  (3, 3), 256, name="conv3_2")
    norm3_3     = norm_layer(conv3_3, name="norm3_3")
    pool3       = max_pool_layer(norm3_3, name="pool3")

    dropout5    = dropout_layer(pool3, 0.5, name="dropout5")

    fc6         = fc_layer(dropout5, 512, name="fc6")
    norm6_1     = norm_layer(fc6, name="norm6_1")    
    dropout6_1  = dropout_layer(norm6_1, 0.5, name="dropout6_1")

    fc7         = fc_layer(dropout6_1, NUM_CLASSES, name="fc7")

    return fc7

def loss(logits, labels):
    """Add L2Loss to all the trainable variables.

    Add summary for "Loss" and "Loss/avg".
    Args:
        logits: Logits from inference().
        labels: Labels from distorted_inputs or inputs(). 1-D tensor
                of shape [batch_size]

    Returns:
        Loss tensor of type float.
    """
    # Calculate the average cross entropy loss across the batch.
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')

def _add_loss_summaries(total_loss):
    """Add summaries for losses in CIFAR-10 model.

    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.

    Args:
        total_loss: Total loss from loss().
    Returns:
        loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.summary.scalar(l.op.name + ' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))

    return loss_averages_op

def train_step(total_loss, global_step):
    """Train CIFAR-10 model.

    Create an optimizer and apply to all trainable variables. Add moving
    average for all trainable variables.

    Args:
        total_loss: Total loss from loss().
        global_step: Integer Variable counting the number of training steps
        processed.
    Returns:
        train_op: op for training.
    """
    # Variables that affect learning rate.
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)
    tf.summary.scalar('learning_rate', lr)

    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = _add_loss_summaries(total_loss)

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.GradientDescentOptimizer(lr)
        grads = opt.compute_gradients(total_loss)

    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    # Add histograms for gradients.
    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)
    with tf.control_dependencies([apply_gradient_op]):
        variables_averages_op = variable_averages.apply(tf.trainable_variables())

    return variables_averages_op


def train():
    with tf.Graph().as_default():
        global_step = tf.train.get_or_create_global_step()
        #Make use of CPU memory to avoid GPU memory allocation problem
        with tf.device('/cpu:0'):  
            #images, labels = cifar10_input.inputs(eval_data=False,
            #                                    data_dir=FLAGS.data_dir,
            #                                    batch_size=FLAGS.batch_size)
            images, labels = cifar10_input.distorted_inputs(data_dir=FLAGS.data_dir,
                                                batch_size=FLAGS.batch_size)

            test_images, test_labels = cifar10_input.inputs(eval_data=True,
                                                data_dir=FLAGS.data_dir,
                                                batch_size=cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL)

        logits = build_vgg_model(images)

        prediction = build_vgg_model(test_images)
        correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.cast(test_labels, tf.int64))
        test_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', test_accuracy)
        summary_op = tf.summary.merge_all()

        # Training
        loss_op = loss(logits, labels)
        train_op = train_step(loss_op, global_step)

        summary_writer = tf.summary.FileWriter(FLAGS.log_dir)
        with tf.train.MonitoredSession() as sess:
            for i in range(FLAGS.max_steps + 1):
                _, loss_val = sess.run([train_op, loss_op])

                if i % FLAGS.log_frequency == 0:
                    test_acc, summary = sess.run([test_accuracy, summary_op])
                    summary_writer.add_summary(summary, i)
                    print("Step:%4d, Loss:%f Test Accuracy:%f"%(i, loss_val, test_acc))

        summary_writer.close()


def main(argv=None):  # pylint: disable=unused-argument
  train()

if __name__ == '__main__':
  tf.app.run()