import tensorflow as tf
import pyaudio
import numpy as np
import os
from voiceRead import ReadVoice

class Trainer():
    def __init__(self, dataDir="training", learning_rate=0.01, training_iter=100):
        self.training_iter = training_iter
        self.dataDir = dataDir
        self.learning_rate = learning_rate

    def getData(self):
        '''loads all wav files within a directory and calls voiceRead functions to normalize them'''
        print "getting data"
        wavs = []
        filenames_list = []
        targets = []
        dir = self.dataDir
        r = ReadVoice()
        # walks the training dir to gather all word names
        for (dirpath, dirnames, filenames) in os.walk(dir):
            filenames_list.extend(filenames)

        for file in filenames:
            print file
            data = r.read_file(self.dataDir + "/" + file)
            wavs.append(data)


        targets = np.array(targets)
        wavs = np.array(wavs)
        print targets.shape, wavs.shape
        return targets, wavs


    def train(self):
        x = tf.placeholder("float", [1024,])
        y = tf.placeholder("float", [3 ])
        W = tf.Variable(tf.zeros([1024,3]))
        b = tf.Variable(tf.zeros([3]))

        with tf.name_scope("Wx-b") as scope:
            model = tf.nn.softmax(tf.matmul(x, W)+b)

        with tf.name_scope("cost_function") as scope:
            cost_function = -tf.reduce_sum(y*tf.log(model))

        with tf.name_scope("train") as scope:
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(cost_function)

        init = tf.initialize_all_variables

        with tf.Session() as sess:
            sess.run(init)
            for iter in range(self.training_iter):
                average_cost = 0.




t = Trainer()

t.getData()