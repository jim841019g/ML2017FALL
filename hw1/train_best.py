import numpy as np
import pandas as pd
import math
import csv

training_data = 'train.csv'
testing_data = 'test.csv'


if __name__ == '__main__':
    
    #preprocess for training data
    traincsv = pd.read_csv(training_data,encoding='Big5')
    traincsv = traincsv.replace('NR',0)
    train_np = traincsv.as_matrix()
    data = train_np[:,2:27]
    data[:,1:26] = data[:,1:26].astype(np.float)
    data = data.reshape((240,18,25))
    pm2_5 = data[:,9,1:26]
    pm10 = data[:,8,1:26]
    pm2_5 = pm2_5.reshape(12,-1)
    pm10 = pm10.reshape(12,-1)
    
    #preprocess for testing data
    text = open(testing_data ,"r")
    row = csv.reader(text , delimiter= ",")
    test_pm25 = []
    test_pm10 = []
    n_row = 0
    for r in row:
        if n_row %18 == 9:
            test_pm25.append([])
            for i in range(2,11):
                test_pm25[int(n_row/18)].append(float(r[i]) )
        elif n_row%18 == 8:
            test_pm10.append([])
            for i in range(2,11):
                test_pm10[int(n_row/18)].append(float(r[i]))
        n_row = n_row+1
    text.close()
    test_pm25 = np.array(test_pm25)
    test_pm10 = np.array(test_pm10)
    
    #start linear regression
    

    y = []
    x_25 = []
    x_10 = []
    for i in range(12):
        for j in range(473):
            y = np.append(y,pm2_5[i][j+7])
            x_25 = np.append(x_25,pm2_5[i][j:j+7])
            x_10 = np.append(x_10,pm10[i][j:j+7])
   
    for i in range(480):
        for j in range(2):
            y = np.append(y,test_pm25[i%240][j+7])
            x_25 = np.append(x_25,test_pm25[i%240][j:j+7])
            x_10 = np.append(x_10,test_pm10[i%240][j:j+7])

    x_25test = []
    x_10test = []
    y_test = []
    for j in range(473):
        x_25test = np.append(x_25test,pm2_5[11][j:j+7])
        x_10test = np.append(x_10test,pm10[11][j:j+7])
        y_test = np.append(y_test,pm2_5[11,j+7])
    
    x_25 = x_25.reshape((-1,7))
    x_10 = x_10.reshape((-1,7))
    x_25test = x_25test.reshape((-1,7))
    x_10test = x_10test.reshape((-1,7))
    
    b = -0.0
    w_25 = np.array([-4.0,-4.0,-4.0,-4.0,-4.0,-4.0,-4.0])
    w_25_2 = np.array([-4.0,-4.0,-4.0,-4.0,-4.0,-4.0,-4.0])
    w_10 = np.array([-4.0,-4.0,-4.0,-4.0,-4.0,-4.0,-4.0])
    w_25 = w_25.reshape((7,1))
    w_25_2 = w_25_2.reshape((7,1))
    w_10 = w_10.reshape((7,1))
    w_10lr = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    b_lr = np.float64(0.0)
    w_25lr = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    w_25_2lr = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0])

    lr = 1
    iteration = 300000
    
    
    for i in range(iteration+1):
        b_grad = 0.0
        w_25grad = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0])
        w_25_2grad = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0])
        w_10grad = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0])
        for n in range(len(x_25)):
            b_grad = b_grad -2.0*(y[n]- b - x_25[n].dot(w_25) - (x_25[n]**2).dot(w_25_2) - x_10[n].dot(w_10))
            w_25grad = w_25grad -2.0*(y[n]- b - x_25[n].dot(w_25) - (x_25[n]**2).dot(w_25_2) - x_10[n].dot(w_10))*x_25[n]
            w_25_2grad = w_25_2grad -2.0*(y[n]- b - x_25[n].dot(w_25) - (x_25[n]**2).dot(w_25_2) - x_10[n].dot(w_10))*(x_25[n]**2)
            w_10grad = w_10grad -2.0*(y[n]- b - x_25[n].dot(w_25)- (x_25[n]**2).dot(w_25_2) - x_10[n].dot(w_10))*x_10[n]
        b_lr = b_lr + b_grad**2
        
        for n in range(7):
            w_25lr[n] = w_25lr[n] + w_25grad[n]**2
            w_25_2lr[n] = w_25_2lr[n] + w_25_2grad[n]**2
            w_10lr[n] = w_10lr[n] + w_10grad[n]**2

        b = b-lr/math.sqrt(b_lr) * b_grad
        for n in range(7):
            w_25[n] = w_25[n]-lr/np.sqrt(w_25lr[n]) * w_25grad[n]
            w_25_2[n] = w_25_2[n]-lr/np.sqrt(w_25_2lr[n]) * w_25_2grad[n]
            w_10[n] = w_10[n]-lr/np.sqrt(w_10lr[n]) * w_10grad[n]
        loss = 0.0
        for n in range(len(x_10)):
            loss += (y[n]- b - x_25[n].dot(w_25)- (x_25[n]**2).dot(w_25_2) - x_10[n].dot(w_10) )**2
        if i%20 == 0:
            loss/= len(x_10)
            print('iteration='+str(i)+', loss='+str(loss))
        if i%2000 == 0:
            output = []
            output = np.append(output,w_25)
            output = np.append(output,w_25_2)
            output = np.append(output,w_10)
            output = np.append(output,b)
            output = np.append(output,w_25lr)
            output = np.append(output,w_25_2lr)
            output = np.append(output,w_10lr)
            output = np.append(output,b_lr)
            output = output.reshape(2,-1)
            name = 'w_b'+str(i)+('.csv')
            f = open(name,'w')
            write = csv.writer(f)
            write.writerows(output)
            f.close()

