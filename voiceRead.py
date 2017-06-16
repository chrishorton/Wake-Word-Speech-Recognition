import pyaudio
from pyaudio import PyAudio
import os
import wave
import numpy as np
import librosa
import tensorflow as tf

class ReadVoice(object):
    def __init__(self, filename=None, format=None, channels=None, rate=None, frames_per_buffer=None, seconds=None, audio=PyAudio()):
        '''FORMAT = pyaudio.paInt16
        CHANNELS = 2
        RATE = 44100
        CHUNK = 1024
        RECORD_SECONDS = 5
        WAVE_OUTPUT_FILENAME = "file.wav"'''
        if filename is None:
            self.filename = filename

        if seconds is None:
            self.seconds = 5
        else:
            self.seconds = seconds
        if format is None:
            self.format = pyaudio.paInt16
        else:
            self.format = format

        if channels is None:
            self.channels = 1
        else:
            self.channels = channels

        if rate is None:
            self.rate = 44100
        else:
            self.rate = rate

        if frames_per_buffer is None:
            self.frames_per_buffer = 1024
        else:
            self.frames_per_buffer = frames_per_buffer

        self.audio = audio

    def dense_to_one_hot(self, batch, batch_size, num_labels):
        sparse_labels = tf.reshape(batch, [batch_size, 1])
        indices = tf.reshape(tf.range(0, batch_size, 1), [batch_size, 1])
        concatenated = tf.concat(1, [indices, sparse_labels])
        concat = tf.concat(0, [[batch_size], [num_labels]])
        output_shape = tf.reshape(concat, [2])
        sparse_to_dense = tf.sparse_to_dense(concatenated, output_shape, 1.0, 0.0)
        return tf.reshape(sparse_to_dense, [batch_size, num_labels])

    def dense_to_one_hot(self, labels_dense, num_classes):
        """Convert class labels from scalars to one-hot vectors."""
        return np.eye(num_classes)[labels_dense]



    def mfcc_batch(self,dataDir="training", batch_size=10):
        batch_features = np.array()
        labels = np.array()

        files = os.listdir(dataDir)



    def mfcc_batch_generator(self, dataDir="training", batch_size=10):
        batch_features = []
        labels = []

        files = os.listdir(dataDir)
        while True:
            print("loaded batch of %d files" % len(files))
            for wav in files:
                wave, sr = librosa.load(dataDir + "/" + wav, mono=True)
                label = self.dense_to_one_hot(int(wav[0]), 3)
                labels.append(label)
                mfcc = librosa.feature.mfcc(wave, sr)
                # print(np.array(mfcc).shape)
                # mfcc = np.pad(mfcc, ((0, 0), (0, 80 - len(mfcc[0]))), mode='constant', constant_values=0)
                np.append(batch_features, np.array(mfcc))
                if len(batch_features) >= batch_size:
                    # print(np.array(batch_features).shape)
                    # yield np.array(batch_features), labels
                    yield batch_features, labels  # basic_rnn_seq2seq inputs must be a sequence
                    batch_features = np.array()  # Reset for next batch
                    labels = np.array()

    def read_mic_stream(self):
        audio = PyAudio()

        stream = audio.open(format=self.format, channels=self.channels,
                            rate=self.rate, input=True,
                            frames_per_buffer=self.frames_per_buffer)
        print "Recording"

        for i in range(0, int(self.rate / self.frames_per_buffer * self.seconds)):
            data = stream.read(self.frames_per_buffer)

        return data



    def convert_stream(self, stream):
        """
        converts stream from readstream to matrix that can be read by tensorflow
        """
        return np.fromstring(stream, "Float32")


    def read_file(self, filename):
        p = PyAudio()
        wf = wave.open(filename, 'rb')
        p.open(format=p.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=wf.getframerate(),
                        input=True,
                        output=True)
        data = wf.readframes(self.frames_per_buffer)
        data = self.convert_stream(data)
        return data

