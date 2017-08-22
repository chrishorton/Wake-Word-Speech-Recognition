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
            # better way to do this (extending it) so that it adds on to the end of the array
            chunk.extend(data)
            data0 = f.readframes(self.CHUNK)
        # finally trim:
        chunk = chunk[0:self.CHUNK * 2]  # should be enough for now -> cut
        chunk.extend(np.zeros(self.CHUNK * 2 - len(chunk)))  # fill with padding 0's
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
        file_names_list = []
        dir = self.dataDir
        print "Generating noise"
        noise = self.gen_gaussian_noise(img_shape)
        # walks the training dir to gather all word names
        for (directory_path, directory_names, file_names) in os.walk(dir):
            file_names_list.extend(file_names)
        for file in file_names_list:
            if file != ".DS_Store":
                print file
                data = self.load_wav_file(self.dataDir + "/" + file)
                wavs.extend([data])
        wavs = np.array(wavs)
        print wavs.shape
        print noise.shape
        np.append(wavs, noise)
        # hack for the labels for now
        targets = np.array([[0], [1]])
        print "Targets", targets
        wavs = wavs.reshape([1, 1024, 8])
        targets.reshape([2,1])
        print "Targets, wavs shapes"
        print targets.shape, wavs.shape
        return targets, wavs

    def tflearn_train(self, targets, wavs):
        #set up input data and place
        net = tflearn.input_data([None, 1024, 8])
        net = tflearn.lstm(net, 128, dropout=0.8)
        net = tflearn.fully_connected(net, 1, activation="softmax")
        net = tflearn.regression(net)

        # what does this do lol - research documentation
        col = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

        for x in col:
            tf.add_to_collection(tf.GraphKeys.VARIABLES, x)
        # builds our model with net variable we assigned all layers to
        model = tflearn.DNN(net, tensorboard_verbose=0)

        while 1:  # training_iters
            model.fit(wavs, targets, n_epoch=10, show_metric=True, batch_size=10)
            # _y = model.predict(wavs)

        model.save("tflearn.lstm.model")
        print (y)




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