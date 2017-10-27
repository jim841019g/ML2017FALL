import sys
import numpy as np
import argparse

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

raw_data = []
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
raw_data = np.concatenate((raw_data,np.ones((raw_data.shape[0],1))), axis=1)

def train():
    data = raw_data / raw_data.max(axis = 0)
    count = 0
    answer = []
    with open(sys.argv[2],'r') as f:
        for linedata in f:
            if count>0:
                answer.append(linedata)
            count+=1
    answer = np.array(answer,dtype = np.int)

    lr = 0.0001
    w = np.ones((107,1))
    iteration = 50000
    batch = 15000
    x = np.zeros((batch,107))
    y = np.zeros((batch,1))
    count = 0
    for i in range (iteration+1):
        for j in range (batch):
            x[j] =data[count]
            y[j] = answer[count]
            count +=1
            if count==data.shape[0]:
                count = 0
        loss = ( sigmoid(x.dot(w)).reshape(1,-1)-y.reshape(1,-1)).dot(x) 
        loss = loss.reshape(-1,1)
        w = w -  lr* loss
        if i%10 == 0:
            pred = sigmoid(data.dot(w))
            correct = 0
            for n in range(pred.size):
                if pred[n]<0.5 and answer[n]==0:
                    correct+=1
                elif pred[n]>0.5 and answer[n]==1:
                    correct+=1
            print (correct)
            print("Epoch: " + str(i) + " - Accuracy: " + str(correct / data.shape[0]))
    np.savetxt('log_w',w)
    return
            
def test():
    #read testdata
    testdata = []
    count = 0
    with open(sys.argv[3],'r') as f:
        for linedata in f:   
            if count>0:
                linedata = linedata.split(",")
                testdata.append(linedata)    
            count+=1
    testdata = np.array(testdata,dtype=np.int)
    testdata = np.concatenate((testdata,np.ones((testdata.shape[0],1))), axis=1)
    test = testdata / raw_data.max(axis = 0)
    w = np.loadtxt('log_w')
    with open(sys.argv[4],'w') as f:
        f.write("id,label\n")
        pred = sigmoid(test.dot(w))
        for i in range(pred.size):
            if pred[i]<0.5:
                f.write(str(i+1)+',0\n')
            else:
                f.write(str(i+1)+',1\n')
    return
                

if __name__ == '__main__':
    #train()
    test()
