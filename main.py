import tensorflow as tf
from voiceRead import ReadVoice

voice = ReadVoice()

voice.convert_stream(voice.read_mic_stream())