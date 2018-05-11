import tensorflow as tf
import cifar10_input
import os

import vgg19_trainable as vgg19

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
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")

NUM_CLASSES = cifar10_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL


with tf.Graph().as_default():
    vgg = vgg19.Vgg19()

    global_step = tf.train.get_or_create_global_step()
    #Make use of CPU memory to avoid GPU memory allocation problem
    #with tf.device('/cpu:0'):
            #images, labels = cifar10_input.inputs(eval_data=False,
            #                                    data_dir=FLAGS.data_dir,
            #                                    batch_size=FLAGS.batch_size)
    train_images, train_labels = cifar10_input.distorted_inputs(data_dir=FLAGS.data_dir,
                                                batch_size=FLAGS.batch_size)

    test_images, test_labels = cifar10_input.inputs(eval_data=True,
                                                data_dir=FLAGS.data_dir,
                                                batch_size=cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL)

    inputs_ = tf.placeholder(tf.float32, [None, 32, 32, 3])
    labels_ = tf.placeholder(tf.int64, [None])

    vgg.build(inputs_)

    cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels_, 
                    logits=vgg.fc8))
    train_step = tf.train.AdamOptimizer().minimize(cross_entropy)

    predicted = tf.nn.softmax(vgg.fc8)
    correct_pred = tf.equal(tf.argmax(predicted, 1), labels_)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    tf.summary.scalar('accuracy', accuracy)
    summary_op = tf.summary.merge_all()
    
    with tf.train.MonitoredSession() as sess:
        images, labels = sess.run([test_images, test_labels])
        test_dict = {inputs_:images, labels_: labels}
        for i in range(FLAGS.max_steps + 1):
            images, labels = sess.run([train_images, train_labels])
            train_dict = {inputs_:images, labels_:labels}

            _, loss = sess.run([train_step, cross_entropy], feed_dict=train_dict)  

            if i % FLAGS.log_frequency == 0:
                test_acc = sess.run(accuracy, feed_dict=test_dict)
                print("Step:%4d Loss:%f Test Accuracy:%f"%(i, loss, test_acc))