import pyaudio
from pyaudio import PyAudio
import wave
import numpy as np

class ReadVoice(object):
    def __init__(self, filename=None, format=None, channels=None, rate=None ,frames_per_buffer=None, seconds=None, audio=PyAudio()):
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

        self.audio = audio


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
        data = np.fromstring(stream, "Float32")
        return data


    def read_file(self, filename):
        p = PyAudio()
        wf = wave.open(filename, 'rb')
        print "stream"
        p.open(format=p.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=wf.getframerate(),
                        input=True,
                        output=True)
        data = wf.readframes(self.frames_per_buffer)
        data = self.convert_stream(data)
        data = data.ravel()
        return data

