import numpy as np
from sklearn import datasets, preprocessing
from neupy import algorithms
import csv
import dill
import math
from math import sqrt
import os
from loss import LossHistory
from sklearn.utils import shuffle
from hyperopt import fmin, tpe, hp, partial, Trials, STATUS_OK, STATUS_FAIL
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, zero_one_loss
import matplotlib.pyplot as plt
import joblib
import pandas as pd
from sklearn.cluster import KMeans
from SOM import Som_simple_zybb as som
import xgboost as xgb
# def SOMcluster():
#     kohonen=algorithms.Kohonen(n_inputs=12, n_outputs=4, step=0.9, verbose=True)
#     kohonen.train(x_train_all, epochs=300)
#     result=kohonen.predict(x_predict)
#     print(result.shape)
#     label_sum=np.sum(result, axis=0)
#     print(label_sum)
#     pd_result=pd.DataFrame(result, columns=['label1', 'label2', 'label3', 'label4'])
#     pd_result.to_csv('C:/Users/chang/Documents/GitHub/dataclean/dataclean/labels.csv')
#     #nonzero_index=np.argmax(result, 1)
#     #pd_label=pd.DataFrame(nonzero_index, columns=['label_index'])
#     #pd_label.to_csv('/home/xcha8737/Downloads/cap/dataclean/index.csv')
#     '''kmeans=KMeans(n_clusters=4, random_state=0)
#     kmeans.fit(X)
#     result = kmeans.predict(X)
#     print(result.shape)
#     label_sum = np.sum(result, axis=0)
#     print(label_sum)
#     pd_result = pd.DataFrame(result, columns=['label1'])
#     pd_result.to_csv('/home/xcha8737/Downloads/cap/dataclean/labels_test.csv')'''
#     #som_zybb = som(category)
#     #som_zybb.initial_output_layer()
#     #som_zybb.som_looper()
#
#
# SOMcluster()


import numpy as np
from sklearn import datasets, preprocessing
from neupy import algorithms
import csv
import dill
import math
from math import sqrt
import os
from loss import LossHistory
from sklearn.utils import shuffle
from hyperopt import fmin, tpe, hp, partial, Trials, STATUS_OK, STATUS_FAIL
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, zero_one_loss,mean_absolute_error,r2_score
import matplotlib.pyplot as plt
import joblib
import pandas as pd
import lightgbm as lgb
from sklearn.cluster import KMeans
import time

#import tensorflow.examples.tutorials.mnist.input_data as input_data

#mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


def dataReader():
    file=open("/home/xcha8737/Downloads/cap/dataclean/all_data.csv", 'r', encoding='utf-8' )
    reader=csv.reader(file)
    features=[]
    output=[]
    for row in reader:
        if row[2]=='ghi':
            continue
        feature=[]
        feature.append(float(row[2]))
        feature.append(float(row[3]))
        feature.append(float(row[4]))
        feature.append(float(row[5]))
        feature.append(float(row[6]))
        feature.append(float(row[7]))
        feature.append(float(row[8]))
        feature.append(float(row[9]))
        feature.append(float(row[10]))
        feature.append(float(row[11]))
        feature.append(float(row[12]))
        feature.append(float(row[13]))
        feature.append(float(row[18]))
        features.append(feature)
        #features.append(row[4:9])
        output.append(float(row[18]))
    file.close()
    X=np.array(features)
    Y=np.array(output)
    return X, Y






def sequence( n_steps):
    x,y=dataReader()
    # x=(X-X.min(axis=0))/(X.max(axis=0)-X.min(axis=0))
    # y=(Y-Y.min(axis=0))/(Y.max(axis=0)-Y.min(axis=0))
    input, output=list(), list()
    #print(x[0:10])
    #print(y[0:10])
    for i in range(len(x)):
        end=i+n_steps
        if end>len(x)-1:
            break
        seq_x, seq_y= x[i:end, :], y[end]
        data_in=np.hstack(seq_x)
        input.append(data_in)
        #input.append(seq_x)
        output.append(seq_y)
    #print(np.shape(input))
    #print(np.shape(output))
    return np.array(input), np.array(output)

# data=pd.read_csv("/home/xcha8737/Desktop/test_data/all_data.csv")
# X=data[["airtemp", "humidity", "insolation", "windspeed", "winddirection"]]
# #X=data[["airtemp", "humidity", "insolation"]]
# Y=data['power (W)']
#X,Y=sequence(9)

def SOMcluster():
    #bach_xs, bach_ys = mnist.train.next_batch(20000)
    #kohonen=algorithms.Kohonen(n_inputs=12, n_outputs=4, step=0.2, verbose=False)
    #kohonen.train(X, epochs=200)
    #result=kohonen.predict(X)
    kmeans=KMeans(n_clusters=5, random_state=0).fit(X)
    result=kmeans.labels_
    result=np.reshape(result, [-1, 1])
    #print(np.reshape(result,[-1,1]))
    #print(result)
    #label_sum=np.sum(result, axis=0)
    #print(label_sum)
    #pd_result=pd.DataFrame(result, columns=['label1', 'label2', 'label3', 'label4'])
    pd_result=pd.DataFrame(result)
    pd_result.to_csv('/home/xcha8737/Downloads/cap/dataclean/labels.csv')
    # nonzero_index=np.argmax(result, 1)
    # pd_label=pd.DataFrame(nonzero_index, columns=['label_index'])
    # pd_label.to_csv('/home/xcha8737/Downloads/cap/dataclean/index.csv')
    joblib.dump(kmeans,'/home/xcha8737/Solar_Forecast/trainning_data/dataclean/dataclean/kmeans.pkl')


# pdata=pd.read_csv('/home/xcha8737/Downloads/cap/dataclean/labels_1.csv')
# data=np.array(pdata)
# X_1=data[:,:-1]
# Y_1=data[:,-1]
# x_1=(X_1-X_1.min(axis=0))/(X_1.max(axis=0)-X_1.min(axis=0))

def SOMcluster1():
    #bach_xs, bach_ys = mnist.train.next_batch(20000)
    #kohonen=algorithms.Kohonen(n_inputs=12, n_outputs=4, step=0.2, verbose=False)
    #kohonen.train(X, epochs=200)
    #result=kohonen.predict(X)
    kmeans=KMeans(n_clusters=2, random_state=0).fit(X_1)
    result=kmeans.labels_
    result=np.reshape(result, [-1, 1])
    #print(np.reshape(result,[-1,1]))
    #print(result)
    #label_sum=np.sum(result, axis=0)
    #print(label_sum)
    #pd_result=pd.DataFrame(result, columns=['label1', 'label2', 'label3', 'label4'])
    pd_result=pd.DataFrame(result)
    #pd_result.to_csv('/home/xcha8737/Downloads/cap/dataclean/labels_3.csv')
    # nonzero_index=np.argmax(result, 1)
    # pd_label=pd.DataFrame(nonzero_index, columns=['label_index'])
    # pd_label.to_csv('/home/xcha8737/Downloads/cap/dataclean/index.csv')
    joblib.dump(kmeans,'/home/xcha8737/Solar_Forecast/trainning_data/dataclean/dataclean/kmeans2.pkl')




# SOMcluster1()
# kmeans=joblib.load('/home/xcha8737/Solar_Forecast/trainning_data/dataclean/dataclean/kmeans2.pkl')
#
#
# #print(X[0].shape)
# #print(kmeans.predict(X[0].reshape([1,-1])))
# # data_temp = X[10].reshape[1, -1]
# # # label = kmeans.predict(data_temp)
# # print(label)
# # pdata=pd.read_csv('/home/xcha8737/Downloads/cap/dataclean/labels_3.csv')
# # data=np.array(pdata)
# # X_1=data[:,:-1]
# # Y_1=data[:,-1]
# # x_1=(X_1-X_1.min(axis=0))/(X_1.max(axis=0)-X_1.min(axis=0))
# data0=[]
# data1=[]
# #data2=[]
# y0=[]
# y1=[]
# #y2=[]
# for i in range(len(X_1)):
#     data_temp=X_1[i].reshape([1,-1])
#     label=kmeans.predict(data_temp)
#     if(label==0):
#         data0.append(X_1[i])
#         y0.append(Y_1[i])
#     # elif(label==1):
#     #     data1.append(X[i])
#     #     y1.append(Y[i])
#     else:
#         data1.append(X_1[i])
#         y1.append(Y_1[i])
# data0=np.array(data0)
# data1=np.array(data1)
# # data2=np.array(data2)
# y0=np.array(y0)
# y1=np.array(y1)
# # y2=np.array(y2)
# y0=y0.reshape([-1,1])
# y1=y1.reshape([-1,1])
# # y2=y2.reshape([-1,1])
# print(y0)
# data0=np.hstack((data0,y0))
# data1=np.hstack((data1,y1))
# # data2=np.hstack((data2,y2))
# pd_result = pd.DataFrame(data0)
# pd_result.to_csv('/home/xcha8737/Downloads/cap/dataclean/labels_11.csv')
# pd_result = pd.DataFrame(data1)
# pd_result.to_csv('/home/xcha8737/Downloads/cap/dataclean/labels_12.csv')
#pd_result = pd.DataFrame(data2)
# pd_result.to_csv('/home/xcha8737/Downloads/cap/dataclean/labels_2.csv')

# SOMcluster()
# kmeans=joblib.load('/home/xcha8737/Solar_Forecast/trainning_data/dataclean/dataclean/kmeans.pkl')
# print(len(X))
#
# #print(X[0].shape)
# #print(kmeans.predict(X[0].reshape([1,-1])))
# # data_temp = X[10].reshape[1, -1]
# # # label = kmeans.predict(data_temp)
# # print(label)
# data0=[]
# data1=[]
# data2=[]
# data3=[]
# data4=[]
# # data5=[]
# y0=[]
# y1=[]
# y2=[]
# y3=[]
# y4=[]
# # y5=[]
# for i in range(len(X)):
#     data_temp=X[i].reshape([1,-1])
#     label=kmeans.predict(data_temp)
#     if(label==0):
#         data0.append(X[i])
#         y0.append(Y[i])
#     elif(label==1):
#         data1.append(X[i])
#         y1.append(Y[i])
#     elif(label==2):
#         data2.append(X[i])
#         y2.append(Y[i])
#     elif(label==3):
#         data3.append(X[i])
#         y3.append(Y[i])
#     # elif(label==4):
#     #     data4.append(X[i])
#     #     y4.append(Y[i])
#     else:
#         data4.append(X[i])
#         y4.append(Y[i])
# data0=np.array(data0)
# data1=np.array(data1)
# data2=np.array(data2)
# data3=np.array(data3)
# data4=np.array(data4)
# # data5=np.array(data5)
# y0=np.array(y0)
# y1=np.array(y1)
# y2=np.array(y2)
# y3=np.array(y3)
# y4=np.array(y4)
# # y5=np.array(y5)
# y0=y0.reshape([-1,1])
# y1=y1.reshape([-1,1])
# y2=y2.reshape([-1,1])
# y3=y3.reshape([-1,1])
# y4=y4.reshape([-1,1])
# # y5=y5.reshape([-1,1])
# print(y0)
# data0=np.hstack((data0,y0))
# data1=np.hstack((data1,y1))
# data2=np.hstack((data2,y2))
# data3=np.hstack((data3,y3))
# data4=np.hstack((data4,y4))
# # data5=np.hstack((data5,y5))
# pd_result = pd.DataFrame(data0)
# pd_result.to_csv('/home/xcha8737/Downloads/cap/dataclean/labels_0.csv')
# pd_result = pd.DataFrame(data1)
# pd_result.to_csv('/home/xcha8737/Downloads/cap/dataclean/labels_1.csv')
# pd_result = pd.DataFrame(data2)
# pd_result.to_csv('/home/xcha8737/Downloads/cap/dataclean/labels_2.csv')
# pd_result = pd.DataFrame(data3)
# pd_result.to_csv('/home/xcha8737/Downloads/cap/dataclean/labels_3.csv')
# pd_result = pd.DataFrame(data4)
# pd_result.to_csv('/home/xcha8737/Downloads/cap/dataclean/labels_4.csv')

# pd_result = pd.DataFrame(data5)
# pd_result.to_csv('/home/xcha8737/Downloads/cap/dataclean/labels_5.csv')


#print(X.shape)
# result=kmeans.predict(X)
# result = np.reshape(result, [-1, 1])
# pd_result = pd.DataFrame(result)
# pd_result.to_csv('/home/xcha8737/Downloads/cap/dataclean/labels1.csv')
#for i in range()

pdata=pd.read_csv('/home/xcha8737/Downloads/cap/dataclean/labels_0.csv')
data0=np.array(pdata)
pdata=pd.read_csv('/home/xcha8737/Downloads/cap/dataclean/labels_1.csv')
data1=np.array(pdata)
pdata=pd.read_csv('/home/xcha8737/Downloads/cap/dataclean/labels_11.csv')
data11=np.array(pdata)
pdata=pd.read_csv('/home/xcha8737/Downloads/cap/dataclean/labels_12.csv')
data12=np.array(pdata)

pdata=pd.read_csv('/home/xcha8737/Downloads/cap/dataclean/labels_2.csv')
data2=np.array(pdata)
pdata=pd.read_csv('/home/xcha8737/Downloads/cap/dataclean/labels_3.csv')
data3=np.array(pdata)
pdata=pd.read_csv('/home/xcha8737/Downloads/cap/dataclean/labels_31.csv')
data31=np.array(pdata)
pdata=pd.read_csv('/home/xcha8737/Downloads/cap/dataclean/labels_32.csv')
data32=np.array(pdata)
pdata=pd.read_csv('/home/xcha8737/Downloads/cap/dataclean/labels_4.csv')
data4=np.array(pdata)

# pdata=pd.read_csv('/home/xcha8737/Downloads/cap/dataclean/labels_5.csv')
# data5=np.array(pdata)


X0=data0[:,:-1]
Y0=data0[:,-1]

X1=data1[:,:-1]
Y1=data1[:,-1]

X11=data11[:,:-1]
Y11=data11[:,-1]
X12=data12[:,:-1]
Y12=data12[:,-1]

X2=data2[:,:-1]
Y2=data2[:,-1]

X3=data3[:,:-1]
Y3=data3[:,-1]
X31=data31[:,:-1]
Y31=data31[:,-1]
X32=data32[:,:-1]
Y32=data32[:,-1]

X4=data4[:,:-1]
Y4=data4[:,-1]

# X5=data5[:,:-1]
# Y5=data5[:,-1]


model0=lgb.Booster(model_file='/home/xcha8737/Solar_Forecast/trainning_data/dataclean/dataclean/GBMmodel_test0001.txt')
# model1=lgb.Booster(model_file='/home/xcha8737/Solar_Forecast/trainning_data/dataclean/dataclean/GBMmodel_test0002.txt')
#model11=lgb.Booster(model_file='/home/xcha8737/Solar_Forecast/trainning_data/dataclean/dataclean/GBMmodel_test00021.txt')
#model12=lgb.Booster(model_file='/home/xcha8737/Solar_Forecast/trainning_data/dataclean/dataclean/GBMmodel_test00022.txt')
model2=lgb.Booster(model_file='/home/xcha8737/Solar_Forecast/trainning_data/dataclean/dataclean/GBMmodel_test0003.txt')
#model3=lgb.Booster(model_file='/home/xcha8737/Solar_Forecast/trainning_data/dataclean/dataclean/GBMmodel_test0004.txt')
# model31=lgb.Booster(model_file='/home/xcha8737/Solar_Forecast/trainning_data/dataclean/dataclean/GBMmodel_test00041.txt')
# model32=lgb.Booster(model_file='/home/xcha8737/Solar_Forecast/trainning_data/dataclean/dataclean/GBMmodel_test00042.txt')
model4=lgb.Booster(model_file='/home/xcha8737/Solar_Forecast/trainning_data/dataclean/dataclean/GBMmodel_test0005.txt')
# # model5=lgb.Booster(model_file='/home/xcha8737/Solar_Forecast/trainning_data/dataclean/dataclean/GBMmodel_test0006.txt')

# model0=xgb.Booster()
# model0.load_model('/home/xcha8737/Solar_Forecast/trainning_data/dataclean/dataclean/xgboost001.model')
model1=xgb.Booster()
model1.load_model('/home/xcha8737/Solar_Forecast/trainning_data/dataclean/dataclean/xgboost002.model')
model11=xgb.Booster()
model11.load_model('/home/xcha8737/Solar_Forecast/trainning_data/dataclean/dataclean/xgboost0021.model')
model12=xgb.Booster()
model12.load_model('/home/xcha8737/Solar_Forecast/trainning_data/dataclean/dataclean/xgboost0022.model')
# model2=xgb.Booster()
# model2.load_model('/home/xcha8737/Solar_Forecast/trainning_data/dataclean/dataclean/xgboost003.model')
model31=xgb.Booster()
model31.load_model('/home/xcha8737/Solar_Forecast/trainning_data/dataclean/dataclean/xgboost0041.model')
model32=xgb.Booster()
model32.load_model('/home/xcha8737/Solar_Forecast/trainning_data/dataclean/dataclean/xgboost0042.model')
# model4=xgb.Booster()
# model4.load_model('/home/xcha8737/Solar_Forecast/trainning_data/dataclean/dataclean/xgboost005.model')



x_train_all0, x_predict0, y_train_all0, y_predict0 = train_test_split(X0, Y0, test_size=0.10, random_state=100)
x_train_all1, x_predict1, y_train_all1, y_predict1 = train_test_split(X1, Y1, test_size=0.10, random_state=100)
x_train_all11, x_predict11, y_train_all11, y_predict11 = train_test_split(X11, Y11, test_size=0.10, random_state=100)
x_train_all12, x_predict12, y_train_all12, y_predict12 = train_test_split(X12, Y12, test_size=0.10, random_state=100)

x_train_all2, x_predict2, y_train_all2, y_predict2 = train_test_split(X2, Y2, test_size=0.10, random_state=100)
x_train_all3, x_predict3, y_train_all3, y_predict3 = train_test_split(X3, Y3, test_size=0.10, random_state=100)
x_train_all31, x_predict31, y_train_all31, y_predict31 = train_test_split(X31, Y31, test_size=0.10, random_state=100)
x_train_all32, x_predict32, y_train_all32, y_predict32 = train_test_split(X32, Y32, test_size=0.10, random_state=100)

x_train_all4, x_predict4, y_train_all4, y_predict4 = train_test_split(X4, Y4, test_size=0.10, random_state=100)
# x_train_all5, x_predict5, y_train_all5, y_predict5 = train_test_split(X5, Y5, test_size=0.10, random_state=100)

# x_predict0 = xgb.DMatrix(x_predict0)
x_predict1 = xgb.DMatrix(x_predict1)
x_predict11 = xgb.DMatrix(x_predict11)
x_predict12 = xgb.DMatrix(x_predict12)
# x_predict2 = xgb.DMatrix(x_predict2)
# x_predict3 = xgb.DMatrix(x_predict3)
x_predict31 = xgb.DMatrix(x_predict31)
x_predict32 = xgb.DMatrix(x_predict32)
# x_predict4 =xgb.DMatrix(x_predict4)

time_start=time.time()
result0=model0.predict(x_predict0)
result1=model1.predict(x_predict1)
# result11=model11.predict(x_predict11)
# result12=model12.predict(x_predict12)
result2=model2.predict(x_predict2)
#result3=model3.predict(x_predict3)
result31=model31.predict(x_predict31)
result32=model32.predict(x_predict32)

result4=model4.predict(x_predict4)
# result5=model5.predict(x_predict5)
time_end=time.time()

result0=result0.reshape(-1,1)
print(result0.shape)
result1=result1.reshape(-1,1)
# result11=result11.reshape(-1,1)
# result12=result12.reshape(-1,1)
result2=result2.reshape(-1,1)
#result3=result3.reshape(-1,1)
result31=result31.reshape(-1,1)
result32=result32.reshape(-1,1)
result4=result4.reshape(-1,1)

result=np.vstack((result0,result1, result2, result31,result32,result4))
#result=np.vstack((result0,result1,result2,result3,result4))
print(result.shape)

y_predict0=y_predict0.reshape(-1,1)
y_predict1=y_predict1.reshape(-1,1)
y_predict11=y_predict11.reshape(-1,1)
y_predict12=y_predict12.reshape(-1,1)
y_predict2=y_predict2.reshape(-1,1)
y_predict3=y_predict3.reshape(-1,1)
y_predict31=y_predict31.reshape(-1,1)
y_predict32=y_predict32.reshape(-1,1)
y_predict4=y_predict4.reshape(-1,1)
# y_predict=np.vstack((y_predict0,y_predict1,y_predict2,y_predict3,y_predict4))

y_predict=np.vstack((y_predict0,y_predict1,y_predict2,y_predict31, y_predict32, y_predict4))
print(y_predict.shape)
# print('y_12 error: ', sqrt(mean_squared_error(y_predict12,result12)))
print('totally cost',time_end-time_start)
print("rmse is ：", sqrt(mean_squared_error(y_predict,result)))
print("mae is ：", mean_absolute_error(y_predict,result))
print('r2 is :', r2_score(y_predict, result))
# com0=mean_squared_error(y_predict0, result0)*len(x_predict0)
# #com1=mean_squared_error(y_predict1, result1)*len(x_predict1)
# com2=mean_squared_error(y_predict2, result2)*len(x_predict2)
# #com3=mean_squared_error(y_predict3, result3)*len(x_predict3)
# com31=mean_squared_error(y_predict31, result31)*len(x_predict31)
# com32=mean_squared_error(y_predict32, result32)*len(x_predict32)
# com4=mean_squared_error(y_predict4, result4)*len(x_predict4)
#com5=mean_squared_error(y_predict5, result5)*len(x_predict5)
# #
# # #mae0=mean_absolute_error(y_predict0, result0)*len(x_predict0)
# print("com0:", np.sqrt(com0/len(x_predict0)) )
#
# print("com1:", np.sqrt(com1/len(x_predict1))  )
#
# print("com2:", np.sqrt(com2/len(x_predict2))  )
#
# #print("com3:", np.sqrt(com3/len(x_predict3))  )
#
# print("com4:", np.sqrt(com4/len(x_predict4))  )
# mse=(com0+com1+com2+com3+com4)/(len(x_predict0)+len(x_predict1)+len(x_predict2)+len(x_predict3)+len(x_predict4))
#
# # mse=(com0+com1+com2+com31+com32+com4)/(len(x_predict0)+len(x_predict1)+len(x_predict2)+len(x_predict31)+len(x_predict32)+len(x_predict4))
# rmse=np.sqrt(mse)
# print('rmse is: ', rmse)

# mae0=mean_absolute_error(y_predict0, result0)*len(x_predict0)
# mae1=mean_absolute_error(y_predict1, result1)*len(x_predict1)
# mae2=mean_absolute_error(y_predict2, result2)*len(x_predict2)
# mae31=mean_absolute_error(y_predict31, result31)*len(x_predict31)
# mae32=mean_absolute_error(y_predict32, result32)*len(x_predict32)
# mae4=mean_absolute_error(y_predict4, result4)*len(x_predict4)
#
# mae=(mae0+mae1+mae2+mae31+mae32+mae4)/(len(x_predict0)+len(x_predict1)+len(x_predict2)+len(x_predict31)+len(x_predict32)+len(x_predict4))
#
# print('mae is: ', mae)







