import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt

from tensorflow.keras.datasets.cifar10 import load_data
from keras.utils import np_utils
import cv2

import timeit
tf.set_random_seed(777)

# hyper parameters
# learning_rate = 0.001
# training_epochs = 100
# batch_size = 100

# def show_image(X):
#     plt.figure(1)
#     k = 0
#     for i in range(0,4):
#         for j in range(0,4):
#             plt.subplot2grid((4,4),(i,j))
#             plt.imshow(X[k], interpolation="bicubic")
#             k = k+1
#     # show the plot
#     plt.show()
#
# def x_data_convertor(x_train,x_test):
#     print(x_train.shape, x_test.dtype)
#     print(x_test.shape, x_test.dtype)
#     x_train = x_train.astype('float32') / 255.0
#     x_test = x_test.astype('float32') / 255.0
#     return x_train, x_test
#
# def y_data_convertor(y_train, y_test):
#     print(y_train.shape, y_train.dtype)
#     print(y_test.shape, y_test.dtype)
#     # y_train = tf.squeeze(tf.one_hot(y_train, 10),axis=1)
#     # y_test = tf.squeeze(tf.one_hot(y_test, 10),axis=1)
#     y_train = np_utils.to_categorical(y_train, 10)
#     y_test = np_utils.to_categorical(y_test, 10)
#     return y_train, y_test
#
def next_batch(num, data, labels):
    idx = np.arange(0, len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]
    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

def get_conv2d(inputs, filters):
    weight_decay = 1e-4
    activation = tf.nn.elu
    return tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=[3, 3], padding="SAME", activation=activation,
                     kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                     kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))

def get_max_pool(inputs):
    return tf.layers.max_pooling2d(inputs=inputs, pool_size=[2, 2], padding="SAME", strides=2)

def get_batch_normalization(inputs):
    return tf.layers.batch_normalization(inputs, center=True, scale=True, training=True)

def get_dropout(inputs, rate):
    return tf.layers.dropout(inputs=inputs, rate=rate, training=is_train)


start = timeit.default_timer();

im = cv2.imread('./data/dog.jpg')
print(im.shape)
img = im.reshape(1, 32, 32, 3)

plt.imshow(im, interpolation="bicubic")
plt.show()
# (x_train, y_train), (x_test, y_test) = load_data()
#
# x_train, x_test = x_data_convertor(x_train, x_test)
# y_train, y_test = y_data_convertor(y_train, y_test)
#
# #show_image(x_test[:16])
#
#train_num_examples = len(x_train)
num_classes = 10
is_train = tf.placeholder(tf.bool)

# input place holders
X = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
Y = tf.placeholder(tf.float32, shape=[None, 10])


conv1 = get_conv2d(X, 32)
norm1 = get_batch_normalization(conv1)
conv1 = get_conv2d(norm1, 32)
norm1 = get_batch_normalization(conv1)
pool1 = get_max_pool(norm1)
drop1 = get_dropout(pool1, 0.2)

conv2 = get_conv2d(drop1, 64)
norm2 = get_batch_normalization(conv2)
conv2 = get_conv2d(norm2, 64)
norm2 = get_batch_normalization(conv2)
pool2 = get_max_pool(norm2)
drop2 = get_dropout(pool2, 0.3)

conv3 = get_conv2d(drop2, 128)
norm3 = get_batch_normalization(conv3)
conv3 = get_conv2d(norm3, 128)
norm3 = get_batch_normalization(conv3)
pool3 = get_max_pool(norm3)
drop3 = get_dropout(pool3, 0.4)

flat = tf.layers.flatten(drop3)
logits = tf.layers.dense(inputs=flat, units=num_classes)

# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y))
# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# initialize
saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, 'model/cifar.ckpt')

    print(
        "Prediction: ",
        sess.run(tf.argmax(logits, 1), feed_dict={X: img, is_train: False})
    )

end = timeit.default_timer()
print(end - start, 'second elapsed...')
