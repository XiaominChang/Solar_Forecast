import joblib
import math
import csv
import keras
import matplotlib.pyplot as plt
from hyperopt import fmin, tpe, hp, partial, Trials, STATUS_OK
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, zero_one_loss,r2_score, mean_absolute_error
import xgboost as xgb
from sklearn.model_selection import train_test_split
import xgboost as xgb
import dill
import datetime
import random
import time
from keras_layer_normalization import LayerNormalization
'''def dataReader():
    file=open("/home/xcha8737/Solar_Forecast/trainning_data/dataclean/dataclean/all_data.csv", 'r', encoding='utf-8' )
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
    X,Y=dataReader()
    print(X.shape)
    print(Y.shape)
    x=(X-X.min(axis=0))/(X.max(axis=0)-X.min(axis=0))
    y=(Y-Y.min(axis=0))/(Y.max(axis=0)-Y.min(axis=0))
    input, output=list(), list()
    #print(x[0:10])
    #print(y[0:10])
    for i in range(len(x)):
        end=i+n_steps
        if end>len(x)-1:
            break
        seq_x, seq_y= x[i:end, :], y[end]
        input.append(seq_x)
        output.append(seq_y)
    #print(np.shape(input))
    #print(np.shape(output))
    return np.array(input), np.array(output)

X,Y=dataReader()
X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
y = (Y - Y.min(axis=0)) / (Y.max(axis=0) - Y.min(axis=0))
#y=Y
n_steps=6
X, y= sequence(n_steps)
y.reshape(-1,1)



xgboost=xgb.Booster()
f=open('/home/xcha8737/Solar_Forecast/trainning_data/dataclean/dataclean/GRNN.dill', 'rb')
xgboost.load_model('/home/xcha8737/Solar_Forecast/trainning_data/dataclean/dataclean/xgboost.model')
grnn=dill.load(f)
svr=joblib.load('/home/xcha8737/Solar_Forecast/trainning_data/dataclean/dataclean/model_svr.pkl')

x_train_all, x_predict, y_train_all, y_predict = train_test_split(X, y, test_size=0.10, random_state=100)
x_train, x_test, y_train, y_test = train_test_split(x_train_all, y_train_all, test_size=0.2, random_state=100)'''

#dpredict = xgb.DMatrix(x_predict)
def dataReader():
    file=open("/home/xcha8737/Solar_Forecast/trainning_data/dataclean/dataclean/all_data.csv", 'r', encoding='utf-8' )
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

X,Y=dataReader()
X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
y = (Y - Y.min(axis=0)) / (Y.max(axis=0) - Y.min(axis=0))
print(X.shape)

x_predict=X[53:102]
y_predict=Y[53:102]
print(y_predict.shape)

def sequence():
    X,Y=dataReader()
    print(X.shape)
    print(Y.shape)
    x=(X-X.min(axis=0))/(X.max(axis=0)-X.min(axis=0))
    y=(Y-Y.min(axis=0))/(Y.max(axis=0)-Y.min(axis=0))
    input, output=list(), list()
    #print(x[0:10])
    #print(y[0:10])
    for i in range(47,96):
        end=i+6
        if end>len(x)-1:
            break
        seq_x, seq_y= x[i:end, :], y[end]
        input.append(seq_x)
        output.append(seq_y)
    #print(np.shape(input))
    #print(np.shape(output))
    return np.array(input), np.array(output)
x_predict,y_in=sequence()
print(x_predict.shape)

lstm=keras.models.load_model('/home/xcha8737/Solar_Forecast/trainning_data/dataclean/dataclean/LSTM.h5', custom_objects={'LayerNormalization':LayerNormalization})
'''time_start=time.time()
result4=lstm.predict(x_predict)
time_end=time.time()
result4=result4.reshape(-1,1)
#y_predict=y_predict.reshape(-1.1)
result4=result4*((Y.max(axis=0) - Y.min(axis=0)))+Y.min(axis=0)
y_predict=y_predict*((Y.max(axis=0) - Y.min(axis=0)))+Y.min(axis=0)
print(mean_squared_error(y_predict, result4))
print(mean_absolute_error(y_predict, result4))
print(np.sqrt(mean_squared_error(y_predict, result4)))
print(r2_score(y_predict,result4))
print('totally cost',time_end-time_start)'''
result4=lstm.predict(x_predict)
result4=result4.reshape(-1,1)
result4=result4*((Y.max(axis=0) - Y.min(axis=0)))+Y.min(axis=0)
x=[i for i in range(49)]
x_ticks=np.arange(0,49, 6)
#x_name=['0am','2am','4am','6am','8am','10am','12pm','2pm','4pm','6pm','8pm','10pm',"0am" ]
x_name=['0am','3am','6am','9am','12pm', '3pm' , '6pm', '9pm', '0am' ]
plt.figure()
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.scatter(x, result4, color='red', linewidth=0.01, alpha=0.75, label='predicton')
plt.scatter(x, y_predict,  color='blue', linewidth=0.01, alpha=0.75, label='actual')
#plt.plot(x, Y[1:50],  color='blue', label='actual')
plt.grid(True, linestyle='--', linewidth=0.5)
plt.xlabel('Time', fontweight='semibold', fontsize='large')
plt.ylabel('PV output (W)', fontweight='semibold',fontsize='large')
plt.xlim((0,49))
plt.xticks(x_ticks, x_name,  fontweight='semibold')
plt.yticks(fontweight='semibold')

plt.legend(loc="upper right")
plt.savefig('/home/xcha8737/Solar_Forecast/trainning_data/dataclean/dataclean/lstm_result.pdf',format = 'pdf', dpi = 1000, bbox_inches = 'tight')

plt.show()
#print(y.shape)
#x_predict=X[1:50]
#y_predict=y[1:50]
#print(y_predict.shape)
'''dpredict = xgb.DMatrix(x_predict)
print("xgboost test: \n")
time_start=time.time()
result1=xgboost.predict(dpredict)
time_end=time.time()
result1=result1.reshape(-1,1)
result1=result1*((Y.max(axis=0) - Y.min(axis=0)))+Y.min(axis=0)
print(mean_squared_error(y_predict, result1))
print(mean_absolute_error(y_predict, result1))
print(np.sqrt(mean_squared_error(y_predict, result1)))
print(r2_score(y_predict,result1))
print('totally cost',time_end-time_start)'''


'''print("grnn test: \n")
time_start=time.time()
result2=grnn.predict(x_predict)
time_end=time.time()
result2=result2*((Y.max(axis=0) - Y.min(axis=0)))+Y.min(axis=0)
print(mean_squared_error(y_predict, result2))
print(mean_absolute_error(y_predict, result2))
print(np.sqrt(mean_squared_error(y_predict, result2)))
print(r2_score(y_predict,result2))
print('totally cost',time_end-time_start)



print("svr test: \n")
time_start=time.time()
result3=svr.predict(x_predict)
time_end=time.time()
result3=result3.reshape(-1,1)
result3=result3*((Y.max(axis=0) - Y.min(axis=0)))+Y.min(axis=0)
print(mean_squared_error(y_predict, result3))
print(mean_absolute_error(y_predict, result3))
print(np.sqrt(mean_squared_error(y_predict, result3)))
print(r2_score(y_predict,result3))
print('totally cost',time_end-time_start)'''




