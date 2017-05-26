import tensorflow as tf
from train import Trainer
from voiceRead import ReadVoice

voice = ReadVoice()
trainer = Trainer()


targets, wavs = trainer.getData()
trainer.train(targets, wavs)