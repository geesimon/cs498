import os
import tensorflow as tf
import numpy as np
from tqdm import tqdm

import cifar10_input
from tensorflow_vgg.vgg16 import Vgg16


BATCH_SIZE  = 100
DATA_DIR = "/tmp/cifar10_data/cifar-10-batches-bin"

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

codes, labels = build_code(DATA_DIR, False, cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN, BATCH_SIZE)
np.save(os.path.join(DATA_DIR, 'train_codes'), codes)
np.save(os.path.join(DATA_DIR, 'train_labels'), labels)

codes, labels = build_code(DATA_DIR, True, cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL, BATCH_SIZE)
np.save(os.path.join(DATA_DIR, 'test_codes'), codes)
np.save(os.path.join(DATA_DIR, 'test_labels'), labels)