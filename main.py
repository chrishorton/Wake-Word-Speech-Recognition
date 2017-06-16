import tensorflow as tf
from train import Trainer
from voiceRead import ReadVoice

voice = ReadVoice()
trainer = Trainer()
batch_size = 64
# TARGETS: 0 = NEXT SLIDE, 1 = PREVIOUS SLIDE, 2 = NOISE
if __name__ == "__main__":

    # targets, wavs = trainer.getData()

    # batch = word_batch = voice.mfcc_batch_generator()
    # X, Y = next(batch)
    # trainX, trainY = X, Y
    # X = X.resahpe
    print "Getting data"
    X, Y = trainer.getData()
    print "Got data"
    trainer.tflearn_train(X, Y)
