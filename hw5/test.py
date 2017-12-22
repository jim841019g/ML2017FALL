import sys
import csv 
import numpy as np
import pandas as pd
from keras.models import Sequential,load_model
from keras.layers import  Activation, Dense,Dropout,Input,Embedding,Flatten,concatenate,add
from keras.callbacks import ModelCheckpoint
from keras.models import Model
import keras.backend.tensorflow_backend as ktf
import tensorflow as tf

def get_session():
    gpu_options = tf.GPUOptions(allow_growth=True)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

ktf.set_session(get_session())


def preprocess():
    training_data = sys.argv[1]
    X_train = pd.read_csv(training_data)
    #print(X_train)
    users = X_train['UserID'].values
    movies = X_train['MovieID'].values
    return users,movies
def test(users,movies):
    model = load_model('best.h5')
    pred = model.predict([movies,users])
    print(pred)
    print(pred.shape)
    with open(sys.argv[2],'w') as f:
        f.write('TestDataID,Rating\n')
        for i in range(len(pred)):
            value = pred[i,0]
            if value>5:
                value = 5.
            elif value<1:
                value = 1.
            f.write(str(i+1)+','+str(value)+'\n')
if __name__ == "__main__":

    users,movies=preprocess()
    test(users,movies)
