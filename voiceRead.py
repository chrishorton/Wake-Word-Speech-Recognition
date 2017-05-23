import pyaudio
from pyaudio import PyAudio

class ReadVoice(object):
    audio = PyAudio()
    def __init__(self, format, channels, rate ,frames_per_buffer, seconds):
        '''FORMAT = pyaudio.paInt16
        CHANNELS = 2
        RATE = 44100
        CHUNK = 1024
        RECORD_SECONDS = 5
        WAVE_OUTPUT_FILENAME = "file.wav"'''
        if seconds is None:
            self.seconds = 5
        else:
            self.seconds = seconds
        if format is None:
            self.format = pyaudio.paInt16
        else:
            self.format = format

        if channels is None:
            self.channels = 2
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

    @property
    def read_stream(self):
        audio = PyAudio()

        stream = audio.open(format=self.format, channels=self.channels,
                            rate=self.rate, input=True,
                            frames_per_buffer=self.chunk)
        frames = []

        for i in range(0, int(self.rate / self.frames_per_buffer * self.seconds)):
            data = stream.read(self.frames_per_buffer)
            frames.append(data)
        return stream

    def convert_stream(self):
        """
        converts stream from readstream to matrix that can be read by tensorflow
        """
        print "Something"