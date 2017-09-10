import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

def createKernel(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def createBias(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def maxPool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


class FashionMNIST:
    def __init__(self):
        self.input = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
        self.output = tf.placeholder(tf.float32, shape=[None, 10])
        self.keep_prob = tf.placeholder(tf.float32)
        #28x28x1
        self.weig1 = createKernel([5, 5, 1, 32])
        self.bias1 = createBias([32])
        self.conv1 = tf.nn.relu(conv2d(self.input, self.weig1) + self.bias1)
        self.pool1 = maxPool(self.conv1)

        #14x14x32
        self.weig2 = createKernel([5, 5, 32, 64])
        self.bias2 = createBias([64])
        self.conv2 = tf.nn.relu(conv2d(self.pool1, self.weig2) + self.bias2)
        self.pool2 = maxPool(self.conv2)

        #7x7x64
        self.weig3 = createKernel([7*7*64, 1024])
        self.bias3 = createBias([1024])
        self.flat  = tf.reshape(self.pool2, [-1, 7*7*64])
        self.fc1   = tf.nn.relu(tf.matmul(self.flat, self.weig3) + self.bias3)
        self.drop = tf.nn.dropout(self.fc1, keep_prob=self.keep_prob)

        #1024
        self.weig4 = createKernel([1024, 10])
        self.bias4 = createBias([10])
        self.fc2 = tf.matmul(self.drop, self.weig4) + self.bias4

        #Loss function
        vars = tf.trainable_variables()
        self.l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in vars])
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.output, logits=self.fc2))

        #Accuracy
        self.correctPrediction = tf.equal(tf.argmax(self.fc2, 1), tf.argmax(self.output, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correctPrediction, tf.float32))

class Reader:
    def __init__(self):
        self.data = input_data.read_data_sets('data')

    def getBatchTrain(self, batchSize):
        batch = self.data.train.next_batch(batchSize)
        images = np.reshape(batch[0], (batchSize, 28, 28, 1))
        labels = np.zeros((batchSize, 10), np.float32)
        for i in xrange(batchSize):
            labels[i][batch[1][i]] = 1
        return images, labels

    def getBatchTest(self, batchSize):
        batch = self.data.test.next_batch(batchSize)
        images = np.reshape(batch[0], (batchSize, 28, 28, 1))
        labels = np.zeros((batchSize, 10), np.float32)
        for i in xrange(batchSize):
            labels[i][batch[1][i]] = 1
        return images, labels


class Solver:
    def __init__(self, _net, _reader, _learningRate, _epochs, _batchSize, _beta):
        self.net = _net
        self.epochs = _epochs
        self.batchSize = _batchSize
        self.reader = _reader
        self.beta = _beta
        self.trainer = tf.train.AdamOptimizer(_learningRate).minimize(self.net.loss + self.net.l2_loss * self.beta)

    def train(self, sess):
        res = 0.0
        sess.run(tf.global_variables_initializer())
        for i in xrange(self.epochs):
            images, labels = reader.getBatchTrain(self.batchSize)
            sess.run(self.trainer, feed_dict = {self.net.input: images, self.net.output: labels, self.net.keep_prob: 0.5})
            loss = sess.run(self.net.loss + self.net.l2_loss * self.beta, feed_dict = {self.net.input: images, self.net.output: labels, self.net.keep_prob: 0.5})

            images, labels = reader.getBatchTest(10000)
            accuracy = sess.run(self.net.accuracy, feed_dict = {self.net.input: images, self.net.output: labels, self.net.keep_prob: 1.0})
            if (accuracy > res):
                res = accuracy
            print 'Iteration %d:' % i, loss, 'Accuracy: ', accuracy


if __name__ == '__main__':
    net = FashionMNIST()
    reader = Reader()
    solver = Solver(net, reader, 1e-4, 50000, 128, 0.0005)
    sess = tf.InteractiveSession()
    accuracy = solver.train(sess)

    print 'Final Accuracy: ', accuracy

