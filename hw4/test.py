import sys
import string
import json
import numpy as np
from gensim.models import Word2Vec
from keras.utils import np_utils
from keras.models import Sequential,load_model
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
    with open(sys.argv[1],'r') as f:
        i=0
        for raw_line in f:
            if i!=0:
                raw_line=raw_line.rstrip('\n')
                raw_line=raw_line.split(",",1)
                linedata = raw_line[1].split(" ")
                raw_data.append(linedata) 
            i+=1
    print(raw_data[0]) 

    n_steps = 25
    model = Word2Vec.load('word2vector.model')
    data = []
    for i in range (len(raw_data)):
        word_temp = raw_data[i]
        temp=np.zeros((n_steps,256))
        j = 0
        for word in word_temp:
            if j <n_steps:
                if word in model.wv.vocab:
                    temp[j]=model.wv[word]
                j+=1
        data.append(temp)
    data = np.array(data,dtype=np.float32)
    return data
def test(data):

    model = load_model('best.h5')
    pred = model.predict(data)
    with open(sys.argv[2],'w') as f:
        f.write('id,label\n')
        for i in range(len(pred)):
            if pred[i]<0.5:
                f.write(str(i)+',0\n')
            else:
                f.write(str(i)+',1\n')
if __name__ =='__main__':
    data = preprocess()
    print(data.shape)
    test(data)
