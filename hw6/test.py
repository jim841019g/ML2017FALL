import sys
import csv 
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from keras.models import Sequential
from keras.layers import  Activation, Dense,Dropout,Input
from keras.callbacks import ModelCheckpoint
from keras.models import Model,load_model
from keras import backend as K
import keras.backend.tensorflow_backend as ktf
import tensorflow as tf

def get_session():
    gpu_options = tf.GPUOptions(allow_growth=True)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

ktf.set_session(get_session())

def predict():
    img_data = np.load(sys.argv[1]).astype(np.float64)
     
    model = load_model('best.h5')
    model.summary()
    #auto_encoder = model.predict(img_data)  
    encoder = K.function([model.layers[0].input],[model.layers[2].output])
    encoder_output = encoder([img_data])[0]
    print(encoder_output[0])
    kmeans = KMeans(n_clusters=2).fit(encoder_output)

    return kmeans.labels_ 

def test(labels):
    #test
    testing_data = sys.argv[2]
    X_test = pd.read_csv(testing_data)
    x1 = X_test['image1_index'].values
    x2 = X_test['image2_index'].values
    with open(sys.argv[3],'w') as f:
        f.write('ID,Ans\n')
        correct=0
        false=0
        for i in range(1980000):
            ans1 = labels[x1[i]]
            ans2 = labels[x2[i]]
            if ans1 == ans2 :
                f.write(str(i)+',1\n')
                correct+=1
            else:
                f.write(str(i)+',0\n')
                false+=1
        print(correct)
        print(false)

if __name__ == '__main__':
    labels = predict()
    test(labels)
