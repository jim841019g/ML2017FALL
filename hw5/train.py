import sys
import csv 
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import  Activation, Dense,Dropout,Input,Embedding,Flatten,concatenate,add,dot
from keras.callbacks import ModelCheckpoint
from keras.models import Model
import keras.backend.tensorflow_backend as ktf
import tensorflow as tf

def get_session():
    gpu_options = tf.GPUOptions(allow_growth=True)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

ktf.set_session(get_session())


def preprocess():
    training_data = 'train.csv'
    X_train = pd.read_csv(training_data)
    sample = X_train.sample(frac=1)
    users = sample['UserID'].values
    movies = sample['MovieID'].values
    ratings = sample['Rating'].values
    return users,movies,ratings
def train(users,movies,ratings):
    
    num_users = max(users)
    num_movies = max(movies)
    print (num_users)
    print (num_movies)
    movie_input = Input(shape=(1,))
    movie_vec = Flatten()(Embedding(num_movies, 32,input_length = 1)(movie_input))
    movie_bias = Flatten()(Embedding(num_movies, 1,input_length = 1)(movie_input))
    movie_vec = Dropout(0.2)(movie_vec)
    user_input = Input(shape=(1,))
    user_vec = Flatten()(Embedding(num_users, 32,input_length = 1)(user_input))
    user_bias = Flatten()(Embedding(num_users, 1,input_length = 1)(user_input))
    user_vec = Dropout(0.2)(user_vec)
    dense = dot([user_vec, movie_vec],axes=-1)
    
    output = add([dense,user_bias,movie_bias])
    model = Model([movie_input,user_input],output)
    model.compile(loss = "mean_squared_error", optimizer = "adamax")
    #model.summary()
    filepath="{epoch:02d}-{val_loss:.4f}-{loss:.4f}.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min', period=1)
    model.fit([movies,users],ratings,epochs=100,validation_split=0.1,callbacks=[checkpoint])

if __name__ == "__main__":

    users,movies,ratings=preprocess()
    train(users,movies,ratings)
