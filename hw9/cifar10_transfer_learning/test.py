import tensorflow as tf

from tensorflow_vgg.vgg16 import Vgg16
import cifar10_input


batch_size  = 100

vgg = Vgg16()

images, labels = cifar10_input.inputs(eval_data=False,
                                        data_dir="/tmp/cifar10_data/cifar-10-batches-bin",
                                        batch_size=batch_size)

vgg.build(images)

#Batch code
global_step = tf.train.get_or_create_global_step()
with tf.train.MonitoredTrainingSession(
    hooks=[tf.train.StopAtStepHook(last_step=cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN // batch_size)]) as sess:    
    while not sess.should_stop():
        codes_batch = sess.run(vgg.relu6)
        print(codes_batch.shape)