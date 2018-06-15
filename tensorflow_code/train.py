import os
import numpy as np
import tensorflow as tf
import input_data
import model
import time
from PIL import Image
import matplotlib.pyplot as plt
import cv2

N_CLASSES = 2
IMG_W = 208
IMG_H = 208
BATCH_SIZE = 16
CAPACITY = 2000
MAX_STEP = 15000  # 至少要在10K以上
learning_rate = 0.0001


def run_training():
    train_dir = 'E:/Code/Dog vs Cat/train/'
    logs_train_dir = 'E:/Code/Dog vs Cat/log/'

    train, train_label = input_data.get_file(train_dir)
    train_batch, train_label_batch = input_data.get_batch(train, train_label,
                                                          IMG_W, IMG_H,
                                                          BATCH_SIZE, CAPACITY)
    train_logits = model.inference(train_batch, BATCH_SIZE, N_CLASSES)
    train_loss = model.losses(train_logits, train_label_batch)
    train_op = model.training(train_loss, learning_rate)
    train_acc = model.evaluation(train_logits, train_label_batch)

    summary_op = tf.summary.merge_all()
    sess = tf.Session()
    train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    time_start =time.time()
    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                break
            _, tra_loss, tra_acc = sess.run([train_op, train_loss, train_acc])
            if step % 50 == 0:
                print('Step %d, train loss = %.2f, train accuracy = %.2f%%' % (step, tra_loss, tra_acc * 100.0))
                summary_str = sess.run(summary_op)
                train_writer.add_summary(summary_str, step)

            if step % 2000 == 0:
                checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

    except tf.errors.OutOfRangeError:
        print('Done training --epoch limit reached')
    finally:
        coord.request_stop()
    time_end = time.time()
    train_time = time_end-time_start
    print("train time:",train_time)
    coord.join(threads)
    sess.close()


def get_one_image(train):
    '''Randomly pick one image from training data
    Return: ndarray
    '''
    n = len(train)
    ind = np.random.randint(0, n)
    img_dir = train[ind]

    image = Image.open('72.jpg')
    # image = plt.imread(img_dir)
    # image = cv2.imread(img_dir)
    # cv2.imshow("input",image)
    # cv2.waitKey(3000)
    plt.imshow(image)
    plt.show()
    image = image.resize([208, 208])
    image = np.array(image)
    return image


def evaluate_one_image():
    '''Test one image against the saved models and parameters
    '''

    # you need to change the directories to yours.
    train_dir = 'E:/Code/Dog vs Cat/test/'
    train = input_data.get_files(train_dir)
    image_array = get_one_image(train)

    with tf.Graph().as_default():
        BATCH_SIZE = 1
        N_CLASSES = 2

        image = tf.cast(image_array, tf.float32)
        image = tf.image.per_image_standardization(image)
        image = tf.reshape(image, [1, 208, 208, 3])

        x = tf.placeholder(tf.float32, shape=[1,208, 208, 3])
        logit = model.inference(x, BATCH_SIZE, N_CLASSES)
        logit = tf.nn.softmax(logit)

        # you need to change the directories to yours.
        logs_train_dir = 'E:/Code/Dog vs Cat/log/'

        saver = tf.train.Saver()

        with tf.Session() as sess:

            print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(logs_train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success, global_step is %s' % global_step)
            else:
                print('No checkpoint file found')

            image_ = sess.run(image)

            prediction = sess.run(logit,feed_dict={x: image_})
            print(prediction)
            max_index = np.argmax(prediction)
            if prediction[:,max_index] > 0.7:
                if max_index == 0:
                    print('This is a cat with possibility %.6f' % prediction[:, 0])
                else:
                    print('This is a dog with possibility %.6f' % prediction[:, 1])
            else:
                print('input error!')

def show_feature():
    log_dir = 'E:/Code/Dog vs Cat/log/'
    image = Image.open('72.jpg')
    plt.imshow(image)
    plt.show()
    image = image.resize([208, 208])
    image = np.array(image)
    image = tf.cast(image, tf.float32)
    image = tf.image.per_image_standardization(image)
    image = tf.reshape(image, [1, 208, 208, 3])

    with tf.variable_scope('conv1'):
        w = tf.get_variable('weights',[3,3,3,16])
        x_min = tf.reduce_min(w)
        x_max = tf.reduce_max(w)
        w_0_to_1= (w - x_min) / (x_max - x_min)
        b = tf.get_variable('biases',[16])
        conv1 = tf.nn.conv2d(image,w_0_to_1,strides=[1,1,1,1],padding='SAME')
        image_b = tf.nn.bias_add(conv1,b)
        image_relu1 = tf.nn.relu(image_b)
    with tf.variable_scope('pooling1_lrn') as scope:
        pool1 = tf.nn.max_pool(image_relu1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pooling1')
        norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001 / 9.0,
                          beta=0.75, name='norm1')
    with tf.variable_scope('conv2'):
        w = tf.get_variable('weights',[3,3,16,16])
        b = tf.get_variable('biases',[16])
        conv2 = tf.nn.conv2d(norm1,w,strides=[1,1,1,1],padding='SAME')
        image_b = tf.nn.bias_add(conv2,b)
        image_relu2 = tf.nn.relu(image_b)

    n_feature = int(image_relu2.get_shape()[-1])
    print (n_feature,image_relu2.get_shape())
    saver = tf.train.Saver(tf.global_variables())
    sess = tf.Session()
    print("Reading checkpoints...")
    ckpt = tf.train.get_checkpoint_state(log_dir)
    if ckpt and ckpt.model_checkpoint_path:
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        saver.restore(sess,ckpt.model_checkpoint_path)
        print('Loading success,global_step is %s' % global_step)
    else:
        print('No checkpoint file found')

    feature_map = tf.reshape(image_relu2,[104,104,n_feature])
    images = sess.run(feature_map)

    plt.figure(figsize=(10, 10))
    for i in np.arange(0, n_feature):
        plt.subplot(4, 4, i + 1)
        plt.axis('off')
        plt.imshow(images[:, :, i])
    plt.show()

if __name__ == '__main__':
    show_feature()
