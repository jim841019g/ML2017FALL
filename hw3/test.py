import csv
import sys
import numpy as np
np.random.seed(1019)
from keras.models import Sequential,load_model 
from keras.utils import np_utils 
from keras import optimizers
import tensorflow as tf
import keras.backend.tensorflow_backend as ktf


def get_session():
    gpu_options = tf.GPUOptions(allow_growth=True)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

ktf.set_session(get_session())

def main():
    
    #testing data
    testing_data = sys.argv[1]
    raw_test = []
    testcsv = csv.DictReader(open(testing_data,'r'),quoting=csv.QUOTE_NONE)

    for row in testcsv:
        raw_test.append(row)

    test = []
    for item in raw_test:
        test.append(item['feature'].split(' '))
    test = np.array(test,dtype=np.int)
    
    model = load_model(sys.argv[3])
    pred = testing(test,model)
    output(pred,sys.argv[2])

def testing(test,model):
    test = test.reshape(test.shape[0], 48, 48, 1)
    test = test/255
    pred = model.predict_classes(test)
    #print(pred)
    return pred

def output(pred,output):
    with open(output,'w') as f:
        f.write('id,label\n')
        for i in range(pred.size):
            f.write(str(i)+','+str(int(pred[i]))+'\n')

if __name__=='__main__':
    main()
