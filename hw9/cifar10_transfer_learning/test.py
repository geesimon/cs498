import tensorflow as tf

from tensorflow_vgg.vgg16 import Vgg16
import cifar10_input
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

batch_size  = 10

vgg = Vgg16()

images, labels = cifar10_input.inputs(eval_data=False,
                                        data_dir="/tmp/cifar10_data/cifar-10-batches-bin",
                                        batch_size=batch_size)

vgg.build(images)

codes = None
#Batch code
global_step = tf.train.get_or_create_global_step()
with tf.train.MonitoredTrainingSession() as sess:
    for i in tqdm(range(cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN // batch_size)):
        codes_batch = sess.run(vgg.relu6)
        if codes is None:
            codes = codes_batch
        else:
            codes = np.concatenate((codes, codes_batch))