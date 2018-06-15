import os
import os.path

import numpy as np
import tensorflow as tf

import input_data
import VGG
import tools
import math
import datetime

N_CLASSES = 2
IMG_W = 208
IMG_H = 208
BATCH_SIZE = 8
CAPACITY = 2000
MAX_STEP = 15000  # 至少要在10K以上
learning_rate = 0.0001
IS_PRETRAIN = True

def train():
    pre_trained_weights = './vgg_weights/vgg16.npy'
    train_dir = 'E:/Code/Dog vs Cat/train/'
    train_log_dir = './logs_vgg/'

    with tf.name_scope('input'):
        image_list, label_list = input_data.get_file(train_dir)
        image_batch,label_batch = input_data.get_batch(image_list, label_list, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)


    x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_H, IMG_W, 3])
    y_ = tf.placeholder(tf.int32, shape=[BATCH_SIZE])

    logits = VGG.VGG16N(x, N_CLASSES, IS_PRETRAIN)
    loss = tools.loss(logits, y_)
    accuracy = tools.accuracy(logits, y_)

    my_global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = tools.optimize(loss, learning_rate, my_global_step)

    saver = tf.train.Saver(tf.global_variables())
    summary_op = tf.summary.merge_all()

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    # load the parameter file, assign the parameters, skip the specific layers
    tools.load_with_skip(pre_trained_weights, sess, ['fc6', 'fc7', 'fc8'])

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    tra_summary_writer = tf.summary.FileWriter(train_log_dir, sess.graph)
    # val_summary_writer = tf.summary.FileWriter(val_log_dir, sess.graph)

    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                break

            tra_images, tra_labels = sess.run([image_batch, label_batch])
            _, tra_loss, tra_acc = sess.run([train_op, loss, accuracy],
                                            feed_dict={x: tra_images, y_: tra_labels})
            if step % 50 == 0 or (step + 1) == MAX_STEP:
                print('Step: %d, loss: %.4f, accuracy: %.4f%%' % (step, tra_loss, tra_acc))
                summary_str = sess.run(summary_op)
                tra_summary_writer.add_summary(summary_str, step)

            # if step % 200 == 0 or (step + 1) == MAX_STEP:
            #     val_images, val_labels = sess.run([val_image_batch, val_label_batch])
            #     val_loss, val_acc = sess.run([loss, accuracy],
            #                                  feed_dict={x: val_images, y_: val_labels})
            #     print('**  Step %d, val loss = %.2f, val accuracy = %.2f%%  **' % (step, val_loss, val_acc))
            #
            #     summary_str = sess.run(summary_op)
                # val_summary_writer.add_summary(summary_str, step)

            if step % 2000 == 0 or (step + 1) == MAX_STEP:
                checkpoint_path = os.path.join(train_log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()


train()