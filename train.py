import tensorflow as tf
import pyaudio
import numpy as np
import os
from voiceRead import ReadVoice
import wave
import tflearn

class Trainer():
    def __init__(self, dataDir="training", learning_rate=0.01, training_iter=100, CHUNK=4096):
        self.training_iter = training_iter
        self.dataDir = dataDir
        self.learning_rate = learning_rate
        self.CHUNK = CHUNK

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""

        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            # self._images = self._images[perm]
            self._image_names = self._image_names[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self.load(self._image_names[start:end]), self._labels[start:end]

    def load_wav_file(self, name):
        f = wave.open(name, "rb")
        # print("loading %s"%name)
        chunk = []
        data0 = f.readframes(self.CHUNK)
        while data0:  # f.getnframes()
            # data=numpy.fromstring(data0, dtype='float32')
            # data = numpy.fromstring(data0, dtype='uint16')
            data = np.fromstring(data0, dtype='uint8')
            data = (data + 128) / 255.  # 0-1 for Better convergence
            # chunks.append(data)
            chunk.extend(data)
            data0 = f.readframes(self.CHUNK)
        # finally trim:
        chunk = chunk[0:self.CHUNK * 2]  # should be enough for now -> cut
        chunk.extend(np.zeros(self.CHUNK * 2 - len(chunk)))  # fill with padding 0's
        # print("%s loaded"%name)
        return chunk

    def gen_gaussian_noise(self, img_shape):
        row, col = img_shape
        mean = 0
        gauss = np.random.normal(mean, 1, (row, col))
        gauss = gauss.reshape(row, col)
        return gauss

    def getData(self):
        '''loads all wav files within a directory and calls voiceRead functions to normalize them'''
        wavs = []
        img_shape = [1024,8]
        filenames_list = []
        targets = []
        dir = self.dataDir
        r = ReadVoice()
        noise = self.gen_gaussian_noise(img_shape)
        # walks the training dir to gather all word names
        for (dirpath, dirnames, filenames) in os.walk(dir):
            filenames_list.extend(filenames)
        index = 0
        for file in filenames:
            targets.append(index)
            index += 1
            print file
            data = self.load_wav_file(self.dataDir + "/" + file)
            # data = r.read_file(self.dataDir+"/"+file)
            wavs.append(data)

        wavs = np.array(wavs)
        np.append(wavs, noise)
        # hack for the labels for now
        targets = np.array([[0], [1]])
        print wavs.shape

        wavs = wavs.reshape([1, 1024, 8])
        print wavs.shape
        targets.reshape([2,1])
        print targets.shape, wavs.shape
        return targets, wavs



    def train(self, targets, wavs):
        x = tf.placeholder("float", [None, 1024])
        y = tf.placeholder("float", [None, 1])
        W = tf.Variable(tf.zeros([1024, 3]))
        b = tf.Variable(tf.zeros([3]))

        with tf.name_scope("Wx-b") as scope:
            model = tf.nn.softmax(tf.matmul(x, W) + b)

        with tf.name_scope("cost_function") as scope:
            cost_function = -tf.reduce_sum(y * tf.log(model))

        with tf.name_scope("train") as scope:
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(cost_function)

        init = tf.initialize_all_variables()

        with tf.Session() as sess:
            sess.run(init)
            average_cost = 0.

            for iter in range(self.training_iter):
                sess.run(optimizer, feed_dict={wavs, targets})
                average_cost += sess.run(cost_function, feed_dict={x: wavs, y: targets})

        predictions = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(predictions, "float"))
        print accuracy


    def tflearn_train(self, targets, wavs):
        net = tflearn.input_data([None, 1024, 8])
        net = tflearn.lstm(net, 128, dropout=0.8)
        net = tflearn.fully_connected(net, 1, activation="softmax")
        net = tflearn.regression(net)

        col = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        for x in col:
            tf.add_to_collection(tf.GraphKeys.VARIABLES, x)

        model = tflearn.DNN(net, tensorboard_verbose=0)

        while 1:  # training_iters
            model.fit(wavs, targets, n_epoch=10, show_metric=True, batch_size=10)
            # _y = model.predict(wavs)

        model.save("tflearn.lstm.model")
        print (y)




