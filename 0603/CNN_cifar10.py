import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt

from tensorflow.keras.datasets.cifar10 import load_data
from keras.utils import np_utils

tf.set_random_seed(777)

# hyper parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100

def show_image(X):
    plt.figure(1)
    k = 0
    for i in range(0,4):
        for j in range(0,4):
            plt.subplot2grid((4,4),(i,j))
            plt.imshow(X[k], interpolation="bicubic")
            k = k+1
    # show the plot
    plt.show()

def x_data_convertor(x_train,x_test):
    print(x_train.shape, x_test.dtype)
    print(x_test.shape, x_test.dtype)

    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    return x_train, x_test

def y_data_convertor(y_train, y_test):
    print(y_train.shape, y_train.dtype)
    print(y_test.shape, y_test.dtype)
    # y_train_one_hot = tf.squeeze(tf.one_hot(y_train, 10),axis=1)
    # y_test_one_hot = tf.squeeze(tf.one_hot(y_test, 10),axis=1)

    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)

    return y_train, y_test

def next_batch(num, data, labels):
    idx = np.arange(0, len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]
    return np.asarray(data_shuffle), np.asarray(labels_shuffle)


(x_train, y_train), (x_test, y_test) = load_data()

x_train, x_test = x_data_convertor(x_train, x_test)
y_train, y_test = y_data_convertor(y_train, y_test)

show_image(x_test[:16])

train_num_examples = len(x_train)
is_train = tf.placeholder(tf.bool)

# input place holders
X = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
Y = tf.placeholder(tf.float32, shape=[None, num_classes])

conv1 = tf.layers.conv2d(inputs=X, filters=32, kernel_size=[3, 3], padding="SAME", activation=tf.nn.relu)
pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], padding="SAME", strides=2)
dropout1 = tf.layers.dropout(inputs=pool1, rate=0.3, training=is_train)

conv2 = tf.layers.conv2d(inputs=dropout1, filters=64, kernel_size=[3, 3], padding="SAME", activation=tf.nn.relu)
pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], padding="SAME", strides=2)
dropout2 = tf.layers.dropout(inputs=pool2, rate=0.3, training=is_train)

conv3 = tf.layers.conv2d(inputs=dropout2, filters=128, kernel_size=[3, 3],  padding="same", activation=tf.nn.relu)
pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], padding="same", strides=2)
dropout3 = tf.layers.dropout(inputs=pool3, rate=0.3, training=is_train)

conv4 = tf.layers.conv2d(inputs=dropout3, filters=256, kernel_size=[3, 3],  padding="same", activation=tf.nn.relu)
pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], padding="same", strides=2)
dropout4 = tf.layers.dropout(inputs=pool4, rate=0.3, training=is_train)

conv5 = tf.layers.conv2d(inputs=dropout4, filters=512, kernel_size=[3, 3],  padding="same", activation=tf.nn.relu)
pool5 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[2, 2], padding="same", strides=2)
dropout5 = tf.layers.dropout(inputs=pool5, rate=0.3, training=is_train)

flat = tf.layers.flatten(dropout5)
dense1 = tf.layers.dense(inputs=flat, units=512, activation=tf.nn.relu)
logits = tf.layers.dense(inputs=dense1, units=10)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# initialize
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # train my model
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(train_num_examples / batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = next_batch(batch_size, x_train, y_train)
            feed_dict = {X: batch_xs, Y: batch_ys, is_train: True}
            c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
            avg_cost += c / total_batch

        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

    print('Learning Finished!')

    # Test model and check accuracy
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print('Accuracy:', sess.run(accuracy, feed_dict={
          X: x_test, Y: y_test, is_train: False}))



