import sys
import numpy as np
import tensorflow as tf
import random
import sklearn
import xgboost 

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

#read train data
raw_data = []
answer = []
title = []
count = 0
with open(sys.argv[1],'r') as f:
    for linedata in f:   
        if count == 0:
            title = linedata.split(",")
        elif count>0:
            linedata = linedata.split(",")
            raw_data.append(linedata)    
        count+=1
raw_data = np.array(raw_data,dtype=np.int)
data = raw_data / raw_data.max(axis = 0)
data = raw_data
#data = data[:,0:63]
count = 0
with open(sys.argv[2],'r') as f:
    for linedata in f:
        if count>0:
            answer.append(linedata)
        count+=1
answer = np.array(answer,dtype = np.int)

model = xgboost.XGBClassifier(max_depth=7, learning_rate=0.1, n_estimators=100,  objective='binary:logistic')
model.fit(data, answer)

pred_x = model.predict(data)
correct = 0
for i in range(pred_x.shape[0]):
    if pred_x[i] == answer[i]:
        correct+=1
print('Accuracy:'+str(correct/pred_x.shape[0]))

#read testdata
testdata = []
count = 0
with open(sys.argv[3],'r') as f:
    for linedata in f:   
        if count>0:
            linedata = linedata.split(",")
            testdata.append(linedata)    
        count+=1
test = np.array(testdata,dtype=np.int)
pred = model.predict(test)
with open(sys.argv[4],'w') as f:
    f.write("id,label\n")
    for i in range(pred.size):
        if pred[i]==0:
            f.write(str(i+1)+',0\n')
        else:
            f.write(str(i+1)+',1\n')
