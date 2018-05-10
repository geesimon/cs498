import os
import tensorflow as tf
import numpy as np
from tqdm import tqdm

import cifar10_input
from tensorflow_vgg.vgg16 import Vgg16

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

def cifar_model(vgg_codes):
    dense = tf.layers.dense(inputs=vgg_codes, units=512, activation=tf.nn.relu)
    dense_nom = tf.layers.batch_normalization(dense)
    dropout = tf.layers.dropout(inputs=dense_nom, rate=0.5)
    logits = tf.layers.dense(dropout, 10, activation=None)

    return logits

def build_code(data_dir, eval_data, sample_count, batch_size = 100):
    vgg = Vgg16()
    with tf.Graph().as_default():
        #global_step = tf.train.get_or_create_global_step()
        with tf.device('/cpu:0'):  #Make use of CPU memory to reduce GPU memory usage
            images, labels = cifar10_input.inputs(eval_data=eval_data,
                                                data_dir=data_dir,
                                                batch_size=batch_size)

        vgg.build(images)

        all_codes = None
        all_labels = None
        with tf.train.MonitoredSession() as sess:
            for _ in tqdm(range(sample_count // batch_size)):    
                labels_batch, codes_batch = sess.run([labels, vgg.relu6])
                if all_codes is None:
                    all_codes = codes_batch
                    all_labels = labels_batch
                else:
                    all_codes = np.concatenate((all_codes, codes_batch))
                    all_labels = np.concatenate((all_labels, labels_batch))

    return all_codes, all_labels

def may_load_test_code(data_dir):
    test_filenames = [os.path.join(data_dir, 'test_codes.npy'), os.path.join(data_dir,'test_labels.npy')]

    #ReBuild test code files if necessary
    if os.path.isfile(test_filenames[0]) and os.path.isfile(test_filenames[1]):
        test_codes = np.load(test_filenames[0])
        test_labels = np.load(test_filenames[1])
    else:
        test_codes, test_labels = build_code(data_dir, True, 10000)
        np.save(test_filenames[0], test_codes)
        np.save(test_filenames[1], test_labels)

    return (test_codes, test_labels)

def train():
    vgg = Vgg16()

    #Test images for VGG only need to be loaded once
    test_codes, test_labels = may_load_test_code(FLAGS.data_dir)

    with tf.Graph().as_default():
        #global_step = tf.train.get_or_create_global_step()
        #Make use of CPU memory to avoid GPU memory allocation problem
        with tf.device('/cpu:0'):  
            #images, labels = cifar10_input.inputs(eval_data=False,
            #                                    data_dir=FLAGS.data_dir,
            #                                    batch_size=FLAGS.batch_size)
            images, labels = cifar10_input.distorted_inputs(data_dir=FLAGS.data_dir,
                                                batch_size=FLAGS.batch_size)

        vgg.build(images)

        with tf.device('/cpu:0'):
            input_ = tf.placeholder(tf.float64, shape=[None, 4096], name = 'cifar-input')
            labels_ = tf.placeholder(tf.int32, [None], name='cifar-output')

        #Append a model for cifar classification
        logits = cifar_model(input_)
        cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels_, 
                    logits=logits))
        train_step = tf.train.AdamOptimizer().minimize(cross_entropy)

        predicted = tf.nn.softmax(logits)
        correct_pred = tf.equal(tf.argmax(predicted, 1, output_type=tf.int32), labels_)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        tf.summary.scalar('accuracy', accuracy)
        summary_op = tf.summary.merge_all()

        summary_writer = tf.summary.FileWriter(FLAGS.log_dir)
        test_dict = {input_:test_codes, labels_:test_labels}
        with tf.train.MonitoredSession() as sess:
            for i in range(FLAGS.max_steps + 1):
                labels_batch, codes_batch = sess.run([labels, vgg.relu6])                
                train_dict = {input_:codes_batch, labels_:labels_batch}
                _, loss_val = sess.run([train_step, cross_entropy], feed_dict=train_dict)

                if i % FLAGS.log_frequency == 0:
                    acc, summary = sess.run([accuracy, summary_op], feed_dict=test_dict)
                    summary_writer.add_summary(summary, i)
                    print("Step:%4d, Loss:%f Test Accuracy:%f"%(i, loss_val, acc))

        summary_writer.close()

def main(argv=None):  # pylint: disable=unused-argument
  train()

if __name__ == '__main__':
  tf.app.run()