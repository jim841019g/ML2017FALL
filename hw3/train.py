import csv
import sys
import numpy as np
np.random.seed(1019)
from keras.models import Sequential,load_model 
from keras.layers import Dense, Dropout, Activation, Flatten 
from keras.layers import Convolution2D, MaxPooling2D ,ZeroPadding2D,AveragePooling2D
from keras.layers.normalization import BatchNormalization 
from keras.utils import np_utils 
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import keras.backend.tensorflow_backend as ktf
from keras.callbacks import ModelCheckpoint


def get_session():
    gpu_options = tf.GPUOptions(allow_growth=True)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

ktf.set_session(get_session())

def main():
    
    epoch = 80    

    #training data
    training_data = sys.argv[1]
    raw_data = []
    traincsv = csv.DictReader(open(training_data,'r'),quoting=csv.QUOTE_NONE)

    for row in traincsv:
        raw_data.append(row)

    raw_label = []
    data = []
    for item in raw_data:
        raw_label.append(item['label'])
        data.append(item['feature'].split(' '))

    raw_label = np.array(raw_label,dtype=np.int)
    label = np.zeros((raw_label.shape[0],7))
    data = np.array(data,dtype=np.int)
    for i in range(raw_label.shape[0]):
        label[i][raw_label[i]]=1

    
    model = training(data,label,epoch)
    #testing(test,model,'answer.csv')
    model.save('model'+str(epoch)+'.h5')
    
def training(data,label,epoch):        
    batch_size = 128
    nb_classes = 7

    #number of convolutional filter
    nb_filters = 64
    #size of pooling area for max pooling
    pool_size = (2,2)
    #convolution kernal size
    kernal_size = (3,3)

    data = data.reshape(data.shape[0], 48, 48, 1)
    rotate = np.rot90(data,axes=(1,2))
    data = np.concatenate((rotate ,data),axis=0)
    data = data /255
    label = np.concatenate((label,label),axis=0) 
        
    model = Sequential()
    
    #cnn1
    model.add(Convolution2D(64,5,5,input_shape = (48,48,1)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(ZeroPadding2D(padding=(2,2)))
    model.add(MaxPooling2D(pool_size=(5,5),strides=(2,2)))
    model.add(ZeroPadding2D(padding=(1,1)))
    model.add(Dropout(0.3))
    #cnn2
    model.add(Convolution2D(64, kernal_size[0], kernal_size[1]))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(ZeroPadding2D(padding=(1,1)))
    model.add(Dropout(0.3))
    #cnn3
    model.add(Convolution2D(128, kernal_size[0], kernal_size[1]))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(AveragePooling2D(pool_size=(3,3),strides=(2,2)))
    model.add(ZeroPadding2D(padding=(1,1)))
    model.add(Dropout(0.3))
    #cnn4
    model.add(Convolution2D(128, kernal_size[0], kernal_size[1]))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(ZeroPadding2D(padding=(1,1)))
    model.add(Dropout(0.3))
    #cnn5
    model.add(Convolution2D(256, kernal_size[0], kernal_size[1]))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(AveragePooling2D(pool_size=(3,3),strides=(2,2)))
    model.add(ZeroPadding2D(padding=(1,1)))
    model.add(Dropout(0.3))
    #dnn1
    model.add(Flatten()) 
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5)) 
    #dnn2
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5)) 
    #dnn3
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5)) 
    #output
    model.add(Dense(nb_classes)) 
    model.add(Activation('softmax'))
    #compile
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    filepath="weights-{epoch:02d}-{val_acc:.4f}.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max', period=1)
    callbacks_list = [checkpoint]

    #training
    model.fit(data[:50000], label[:50000], batch_size=batch_size, nb_epoch=epoch,verbose=1,validation_data=(data[50000:],label[50000:]))
    return model

def testing(test,model,output):
    test = test.reshape(test.shape[0], 48, 48, 1)
    test = test/255
    pred = model.predict_classes(test)
    #print(pred)
    
    with open(output,'w') as f:
        f.write('id,label\n')
        for i in range(pred.size):
            f.write(str(i)+','+str(pred[i])+'\n')

if __name__=='__main__':
    main()
