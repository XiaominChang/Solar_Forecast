from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import math
import csv
from tensorflow import keras
import matplotlib.pyplot as plt
from hyperopt import fmin, tpe, hp, partial, Trials, STATUS_OK
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, zero_one_loss,r2_score
import xgboost as xgb
from sklearn.model_selection import train_test_split
import xgboost as xgb
import dill
import datetime
import random
from keras_layer_normalization import LayerNormalization
import lightgbm as lgb
import time
import pandas as pd



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
        #feature.append(float(row[18]))

        features.append(feature)
        #features.append(row[4:9])
        output.append(float(row[18]))
    file.close()
    X=np.array(features)
    Y=np.array(output)
    return X, Y


'''def sequence( n_steps):
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
        data_in=np.hstack(seq_x)
        input.append(data_in)
        output.append(seq_y)
    #print(np.shape(input))
    #print(np.shape(output))
    return np.array(input), np.array(output)'''

'''X,Y=dataReader()
X1=X
X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
y = (Y - Y.min(axis=0)) / (Y.max(axis=0) - Y.min(axis=0))
print(X.shape)
xgboost=xgb.Booster()
#f=open('/home/xcha8737/Solar_Forecast/trainning_data/dataclean/dataclean/GRNN.dill', 'rb')
#xgboost.load_model('/home/xcha8737/Solar_Forecast/trainning_data/dataclean/dataclean/xgboost.model')

xgboost.load_model('/home/xcha8737/env/vagrant_cluster/Forecast/xgboost_para.model')
#grnn=dill.load(f)
#gru=keras.models.load_model('C:/Users/chang/Documents/GitHub/dataclean/dataclean/GRU.h5')
#svr=joblib.load('/home/xcha8737/Solar_Forecast/trainning_data/dataclean/dataclean/model_svr.pkl')

x_train_all, x_predict, y_train_all, y_predict = train_test_split(X1, Y, test_size=0.10, random_state=100)
x_train, x_test, y_train, y_test = train_test_split(x_train_all, y_train_all, test_size=0.2, random_state=100)
dpredict = xgb.DMatrix(x_predict)
#n=y_predict.size
#x_predict=X[53:102]
#x_predict=x_predict*((X1.max(axis=0) - X1.min(axis=0)))+X1.min(axis=0)
#y_predict=Y[53:102]
print(y_predict.shape)'''

'''def sequence():
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
print(x_predict.shape)'''
'''lstm=keras.models.load_model('/home/xcha8737/Solar_Forecast/trainning_data/dataclean/dataclean/LSTM.h5', custom_objects={'LayerNormalization':LayerNormalization})

result=lstm.predict(x_predict)
result=result*((Y.max(axis=0) - Y.min(axis=0)))+Y.min(axis=0)
x=[i for i in range(49)]
x_ticks=np.arange(0,49, 6)
#x_name=['0am','2am','4am','6am','8am','10am','12pm','2pm','4pm','6pm','8pm','10pm',"0am" ]
x_name=['0am','3am','6am','9am','12pm', '3pm' , '6pm', '9pm', '0am' ]
plt.figure()
plt.scatter(x, result, color='red', linewidth=0.01, alpha=0.75, label='predicton')
plt.scatter(x, y_predict,  color='blue', linewidth=0.01, alpha=0.75, label='actual')
#plt.plot(x, Y[1:50],  color='blue', label='actual')
plt.grid(True, linestyle='--', linewidth=0.5)
plt.xlabel('Time', fontweight='semibold', fontsize='large')
plt.ylabel('PV output (W)', fontweight='semibold',fontsize='large')
plt.xlim((0,49))
plt.xticks(x_ticks, x_name,  fontweight='semibold')
plt.yticks(fontweight='semibold')
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.legend(loc="upper right")
plt.savefig('/home/xcha8737/Solar_Forecast/trainning_data/dataclean/dataclean/lstm_result.pdf',format = 'pdf', dpi = 1000, bbox_inches = 'tight')

plt.show()'''

# data=pd.read_csv("/home/xcha8737/Desktop/test_data/all_data.csv")
# X=data[["airtemp", "humidity", "insolation", "windspeed", "winddirection"]]
# #X=data[["airtemp", "humidity", "insolation"]]
# Y=data['power (W)']

xgboost=xgb.Booster()
xgboost.load_model('/home/xcha8737/Solar_Forecast/trainning_data/dataclean/dataclean/xgboost.model')


x_train_all, x_predict, y_train_all, y_predict = train_test_split(X, Y, test_size=0.10, random_state=100)

x_train, x_test, y_train, y_test = train_test_split(x_train_all, y_train_all, test_size=0.2, random_state=100)

#lgbm=lgb.Booster(model_file='/home/xcha8737/env/vagrant_cluster/Forecast/LightGBM_model_parallel2.txt')
#lgbm=lgb.Booster(model_file='/home/xcha8737/Solar_Forecast/trainning_data/dataclean/dataclean/GBMmodel_test1100.txt')
dpredict = lgb.Dataset(x_predict)
time_start=time.time()
#result1=lgbm.predict(x_predict)
result1=xgboost.predict(dpredict)
time_end=time.time()
#result1=result1*((Y.max(axis=0) - Y.min(axis=0)))+Y.min(axis=0)
#y_predict=y_predict*((Y.max(axis=0) - Y.min(axis=0)))+Y.min(axis=0)
print(mean_squared_error(y_predict, result1))
print(mean_absolute_error(y_predict, result1))
print(np.sqrt(mean_squared_error(y_predict, result1)))
print(r2_score(y_predict,result1))
print('totally cost',time_end-time_start)

'''x=[i for i in range(49)]
x_ticks=np.arange(0,49, 6)
#x_name=['0am','2am','4am','6am','8am','10am','12pm','2pm','4pm','6pm','8pm','10pm',"0am" ]
x_name=['0am','3am','6am','9am','12pm', '3pm' , '6pm', '9pm', '0am' ]


plt.figure()
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.scatter(x, result1, color='red', linewidth=0.01, alpha=0.75, label='predicton')
plt.scatter(x, y_predict,  color='blue', linewidth=0.01, alpha=0.75, label='actual')
#plt.plot(x, Y[1:50],  color='blue', label='actual')
plt.grid(True, linestyle='--', linewidth=0.5)
plt.xlabel('Time', fontweight='semibold', fontsize='large')
plt.ylabel('PV output (W)', fontweight='semibold',fontsize='large')
plt.xlim((0,49))
plt.xticks(x_ticks, x_name,  fontweight='semibold')
plt.yticks(fontweight='semibold')

plt.legend(loc="upper right")




plt.savefig('/home/xcha8737/Solar_Forecast/trainning_data/dataclean/dataclean/xgboost_result.pdf',format = 'pdf', dpi = 1000, bbox_inches = 'tight')

plt.show()'''





'''dpredict = xgb.DMatrix(x_predict)

result1=xgboost.predict(dpredict)
result2=grnn.predict(x_predict)
result3=svr.predict(x_predict)
result1=result1*((Y.max(axis=0) - Y.min(axis=0)))+Y.min(axis=0)
result2=result2*((Y.max(axis=0) - Y.min(axis=0)))+Y.min(axis=0)
result3=result3*((Y.max(axis=0) - Y.min(axis=0)))+Y.min(axis=0)
#y_predict=y_predict*((Y.max(axis=0) - Y.min(axis=0)))+Y.min(axis=0)

x=[i for i in range(49)]
x_ticks=np.arange(0,49, 6)
#x_name=['0am','2am','4am','6am','8am','10am','12pm','2pm','4pm','6pm','8pm','10pm',"0am" ]
x_name=['0am','3am','6am','9am','12pm', '3pm' , '6pm', '9pm', '0am' ]


plt.figure()
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.scatter(x, result1, color='red', linewidth=0.01, alpha=0.75, label='predicton')
plt.scatter(x, y_predict,  color='blue', linewidth=0.01, alpha=0.75, label='actual')
#plt.plot(x, Y[1:50],  color='blue', label='actual')
plt.grid(True, linestyle='--', linewidth=0.5)
plt.xlabel('Time', fontweight='semibold', fontsize='large')
plt.ylabel('PV output (W)', fontweight='semibold',fontsize='large')
plt.xlim((0,49))
plt.xticks(x_ticks, x_name,  fontweight='semibold')
plt.yticks(fontweight='semibold')

plt.legend(loc="upper right")


print(mean_squared_error(y_predict, result1))

plt.savefig('/home/xcha8737/Solar_Forecast/trainning_data/dataclean/dataclean/xgboost_result.pdf',format = 'pdf', dpi = 1000, bbox_inches = 'tight')

plt.show()


plt.figure()
plt.scatter(x, result2, color='red', linewidth=0.01, alpha=0.75, label='predicton')
plt.scatter(x, y_predict,  color='blue', linewidth=0.01, alpha=0.75, label='actual')
#plt.plot(x, Y[1:50],  color='blue', label='actual')
plt.grid(True, linestyle='--', linewidth=0.5)
plt.xlabel('Time', fontweight='semibold', fontsize='large')
plt.ylabel('PV output (W)', fontweight='semibold',fontsize='large')
plt.xlim((0,49))
plt.xticks(x_ticks, x_name,  fontweight='semibold')
plt.yticks(fontweight='semibold')
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.legend(loc="upper right")


print(mean_squared_error(y_predict, result1))

plt.savefig('/home/xcha8737/Solar_Forecast/trainning_data/dataclean/dataclean/grnn_result.pdf',format = 'pdf', dpi = 1000, bbox_inches = 'tight')

plt.show()


plt.figure()
plt.scatter(x, result3, color='red', linewidth=0.01, alpha=0.75, label='predicton')
plt.scatter(x, y_predict,  color='blue', linewidth=0.01, alpha=0.75, label='actual')
#plt.plot(x, Y[1:50],  color='blue', label='actual')
plt.grid(True, linestyle='--', linewidth=0.5)
plt.xlabel('Time', fontweight='semibold', fontsize='large')
plt.ylabel('PV output (W)', fontweight='semibold',fontsize='large')
plt.xlim((0,49))
plt.xticks(x_ticks, x_name,  fontweight='semibold')
plt.yticks(fontweight='semibold')
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.legend(loc="upper right")


print(mean_squared_error(y_predict, result1))

plt.savefig('/home/xcha8737/Solar_Forecast/trainning_data/dataclean/dataclean/svr_result.pdf',format = 'pdf', dpi = 1000, bbox_inches = 'tight')

plt.show()'''

'''result2=grnn.predict(x_predict)
print(result2.shape)
print(mean_squared_error(y_predict, result2))

result3=svr.predict(x_predict)
print(result3.shape)
print(mean_squared_error(y_predict, result3))
result3=result3.reshape(-1,1)


result_predict=np.hstack([result1,result2,result3])

print(result_predict.shape)


mixed=keras.models.load_model('/home/xcha8737/Solar_Forecast/trainning_data/dataclean/dataclean/mixed_model.h5')
result4=mixed.predict(result_predict)
print(mean_squared_error(y_predict, result4))'''







'''result1=result1.reshape(-1,1)
result1=result1*((Y.max(axis=0) - Y.min(axis=0)))+Y.min(axis=0)
x=[i for i in range(49)]
x_ticks=['0:00','0:30','1:00','1:30','2:00','2:30','3:00','3:30','4:00','4:30','5:00','5:30','6:00','6:30','7:00','7:30','8:00','8:30','9:00','9:30','10:00','10:30','11:00','11:30','12:00','12:30','13:00','13:30','14:00','14:30','15:00','15:30','16:00','16:30','17:00','17:30','18:00','18:30','19:00','19:30','20:00','20:30','21:00','21:30','22:00','22:30','23:00','23:30',"00:00" ]
#x=np.array(x)
#x = [ datetime.datetime.now()+datetime.timedelta(hours=i) for i in range(49)]
x_ticks=np.arange(0, 25, 0.5)

plt.figure()
plt.scatter(x, result1, color='red', linewidth=0.01, alpha=0.75, label='predicton')
plt.scatter(x, Y[1:50],  color='blue', linewidth=0.01, alpha=0.75, label='actual')
#plt.plot(x, Y[1:50],  color='blue', label='actual')
#plt.grid(True)
plt.xlabel('time')
plt.ylabel('output')
#plt.xlim((0,50))
plt.xticks(x_ticks, rotation=5)
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.legend(loc="upper right")
plt.show()'''

