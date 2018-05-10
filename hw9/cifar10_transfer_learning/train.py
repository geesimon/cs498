import os
import tensorflow as tf
import numpy as np

max_steps = 20000
batch_size = 1000
keep_prob = 0.4

class CIFAR10_VGG16_Code:
    def __init__(self, data_dir = '/tmp/cifar10_data/cifar-10-batches-bin'):
        train_filenames = ['train_codes.npy', 'train_labels.npy']
        test_filenames = ['test_codes.npy', 'test_labels.npy']
        
        self.train_start = 0
        self.test_start = 0
        
        self.train_codes = np.load(os.path.join(data_dir, train_filenames[0]))
        self.train_labels = np.load(os.path.join(data_dir, train_filenames[1]))
        self.test_codes = np.load(os.path.join(data_dir, test_filenames[0]))
        self.test_labels = np.load(os.path.join(data_dir, test_filenames[1]))
    
    def next_batch(self, eval_data = False, batch_size = 100):
        batch_codes = np.zeros((batch_size, self.train_codes.shape[1]))
        batch_labels = np.int32(np.zeros(batch_size))
        
        if eval_data :
            codes = self.test_codes
            labels = self.test_labels
            start = self.test_start
        else:
            codes = self.train_codes
            labels = self.train_labels
            start = self.train_start
        
        for i in range(batch_size):
            batch_codes[i] = codes[start]
            batch_labels[i] = labels[start]
            
            start += 1
            if start >= codes.shape[0]:
                start = 0
        
        if eval_data :
            self.test_start = start
        else:
            self.train_start = start
                
        return (batch_codes, batch_labels)

    def test_data(self):
        return (self.test_codes, self.test_labels)

vgg16_dataset = CIFAR10_VGG16_Code()

with tf.Graph().as_default():
    with tf.device('/cpu:0'):
        inputs_ = tf.placeholder(tf.float64, shape=[None, vgg16_dataset.train_codes.shape[1]], name = 'x-input')
        labels_ = tf.placeholder(tf.int32, [None], name='y-input')

<<<<<<< HEAD
dense = tf.contrib.layers.fully_connected(inputs_, 1024)
#dense = tf.layers.dense(inputs=inputs_, units=1024, activation=tf.nn.relu)
dropout = tf.layers.dropout(inputs=dense, rate=keep_prob)
logits = tf.layers.dense(dropout, 10, activation=None)
=======
    #fc = tf.contrib.layers.fully_connected(inputs_, 256)
    #logits = tf.contrib.layers.fully_connected(fc, 10, activation_fn=None)
    dense = tf.layers.dense(inputs=inputs_, units=512, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate=keep_prob)
    logits = tf.layers.dense(dropout, 10, activation=None)

    cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels_, 
                        logits=logits))
>>>>>>> 67d9908100271c48df3aa50ee9037564c3faeaba

    train_step = tf.train.AdamOptimizer().minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.cast(labels_, tf.int64))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        test_dict = {inputs_:vgg16_dataset.test_codes, labels_:vgg16_dataset.test_labels}
        for i in range(max_steps + 1):
            train_codes, train_labels = vgg16_dataset.next_batch()
            train_dict = {inputs_:train_codes, labels_:train_labels}

<<<<<<< HEAD
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(max_steps + 1):
        train_codes, train_labels = vgg16_dataset.next_batch()
        train_dict = {inputs_:train_codes, labels_:train_labels}
        #test_dict = {inputs_:mnist.test.images, targets_:mnist.test.labels, keep_prob:1}
        _, loss_val = sess.run([train_step, cross_entropy], feed_dict=train_dict)
=======
            _, loss_val = sess.run([train_step, cross_entropy], feed_dict=train_dict)
>>>>>>> 67d9908100271c48df3aa50ee9037564c3faeaba

            if i % 100 == 0:
                acc = sess.run(accuracy, feed_dict=test_dict)
                print("Step %3d: Accuracy:%f"%(i, acc))