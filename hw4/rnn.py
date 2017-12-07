import sys
import string
import json
import numpy as np
from gensim.models import Word2Vec
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import SimpleRNN, Activation, Dense,Dropout,LSTM
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint
import keras.backend.tensorflow_backend as ktf
import tensorflow as tf

def get_session():
    gpu_options = tf.GPUOptions(allow_growth=True)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

ktf.set_session(get_session())


def preprocess():
    
    raw_data = []
    label = []
    with open(sys.argv[1],'r') as f:
        for raw_line in f:
            raw_line=raw_line.rstrip('\n')
            raw_line=raw_line.split(" +++$+++ ")
            label.append(raw_line[0])
            linedata = raw_line[1].split(" ")
            raw_data.append(linedata) 
    label = np.array(label,dtype=np.int)
    

    n_steps = 25
    model = Word2Vec.load('word2vector.model')
    data = []
    for i in range (len(raw_data)):
        word_temp = raw_data[i]
        temp=np.zeros((n_steps,256))
        j = 0
        for word in word_temp:
            if j <25:
                if word in model.wv.vocab:
                    temp[j]=model.wv[word]
                j+=1
        data.append(temp)
    data = np.array(data,dtype=np.float32)
    return label,data

def train(label,data):
    n_steps = 25
    n_input = 256
    batch_size = 128
    epoch = 50
    val = 190000
    x_train = data[:val]
    x_val = data[val:]
    y_train = label[:val]
    y_val = label[val:]
    print(x_train.shape)
    print(y_train.shape)
    print(x_val.shape)
    print(y_val.shape)

    model = Sequential()
    model.add( LSTM(64,dropout=0.4,input_shape=(n_steps,n_input) ))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Dropout(0.25))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    
    filepath="{epoch:02d}-{val_acc:.4f}.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max', period=1)
    
    model.fit(x_train,y_train,batch_size=batch_size,nb_epoch=epoch,validation_data=(x_val,y_val),callbacks=[checkpoint])

if __name__ =='__main__':
    label,data = preprocess()
    train(label,data)
