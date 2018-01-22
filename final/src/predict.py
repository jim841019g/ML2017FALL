# from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import argparse
import json
import os

import sys

from keras import backend as K
from keras.models import Model, load_model

from layers import Argmax
from data import BatchGen, load_dataset
from utils import custom_objects
import pickle

np.random.seed(10)

id_list = pickle.load(open('test_id','rb'))
start_list = pickle.load(open('test_start','rb'))
end_list = pickle.load(open('test_end','rb'))

model = '89.model'
prediction = sys.argv[1]

print('Preparing model...', end='')
model = load_model(model, custom_objects())

inputs = model.inputs
outputs = [ Argmax() (output) for output in model.outputs ]

predicting_model = Model(inputs, outputs)
print('Done!')

print('Loading data...', end='')
dev_data = load_dataset('test_data')
char_level_embeddings = len(dev_data[0]) is 4
maxlen = [250,250,30,30] if char_level_embeddings else [250, 30]
dev_data_gen = BatchGen(*dev_data, batch_size=70, shuffle=False, group=False, maxlen=maxlen)


print('Running predicting model...', end='')
predictions = predicting_model.predict_generator(generator=dev_data_gen,
                                                 steps=dev_data_gen.steps(),
                                                 verbose=1)
print('Done!')
print(len(predictions[0]))
with open(prediction,'w') as f:
    f.write('id,answer\n')
    for i in range(len(predictions[0])):
        sub = predictions[1][i] - predictions[0][i]
        f.write(id_list[i]+',')
        start = predictions[0][i]+start_list[i]
        end = min(predictions[1][i]+1+start_list[i],end_list[i])
        for j in range(start,end):
            f.write(str(j)+' ')
        f.write('\n')
        #else:
        #    f.write(ylc_list[i+1])
