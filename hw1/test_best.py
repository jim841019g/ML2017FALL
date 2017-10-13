import numpy as np
import pandas as pd
import math
import csv
import sys

testing_data = sys.argv[1]
output_data = sys.argv[2]

if __name__ == '__main__':
    
    #preprocess
    text = open(testing_data ,"r")
    row = csv.reader(text , delimiter= ",")
    testcsv = []
    n_row = 0
    for r in row:
        testcsv.append([])
        for i in range(2,11):
            if r[i] !="NR":
                testcsv[n_row].append(float(r[i]))
            else:
                testcsv[n_row].append(0)
        n_row = n_row+1
    text.close()
    data = np.array(testcsv)
    data = data.reshape((240,18,-1))
    #print (data)
    
    pm2_5 = data[:,9,2:9]
    pm10 = data[:,8,2:9]
    #print (pm2_5)
    #print (pm10)
    
    #start linear regression
    csvname = 'w_b300000.csv'
    w_b_csv = csv.reader(open(csvname,'r'))
    w_b = []
    for row in w_b_csv:
        w_b = np.append(w_b,row)
    w_b = w_b.reshape(2,-1)
    b = float(w_b[0,21])
    w_25 = []
    w_25_2 = []
    w_10 = []
    for i in range(7):
        w_25 = np.append(w_25,float(w_b[0,i]))
        w_25_2 = np.append(w_25_2,float(w_b[0,i+7]))
        w_10 = np.append(w_10,float(w_b[0,i+14]))
    outputdata = []
    outputdata.append([])
    outputdata[0] = ['id','value']
    for j in range(len(pm2_5)):
        outputdata.append([])
        ans = b + pm2_5[j].dot(w_25) + (pm2_5[j]**2).dot(w_25_2) + pm10[j].dot(w_10)
        outputdata[j+1].append('id_'+str(j))
        outputdata[j+1].append(ans)
        #print (ans)
    f = open(output_data,'w')
    write = csv.writer(f)
    write.writerows(outputdata)
    f.close()
    #a = [0,1,2,3,4,5,6,7,8,9,10]
