import tensorflow as tf
import pyaudio
import wave
import os
from voiceRead import ReadVoice

class Trainer():
    def __init__(self, dataDir="training"):
        self.dataDir = dataDir

    def getData(self):
        '''loads all wav files within a directory and calls voiceRead functions to normalize them'''
        print "getting data"
        wavs = []
        filenames_list = []
        targets = []
        dir = self.dataDir
        p = pyaudio.PyAudio()
        r = ReadVoice()
        # walks the training dir to gather all word names
        for (dirpath, dirnames, filenames) in os.walk(dir):
            filenames_list.extend(dirnames)

        for directory in filenames_list:
            for (dirpath,dirnames,filenames) in os.walk(dir + "/" + directory):
                times_run = 0
                if times_run == 0:
                    targets.extend(directory * len(filenames))

                times_run+=1

                for file in filenames:
                    data = r.read_file(file)
                    wavs.append(data)




    def train(self):
        print "training..."