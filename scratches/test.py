from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import math
import csv
from tensorflow import keras
import matplotlib.pyplot as plt
from hyperopt import fmin, tpe, hp, partial, Trials, STATUS_OK
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, zero_one_loss
import xgboost as xgb
from sklearn.model_selection import train_test_split
import xgboost as xgb
import dill
def dataReader():
    file=open("C:/Users/chang/Documents/GitHub/dataclean/dataclean/all_data.csv", 'r', encoding='utf-8' )
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
        data_in=np.hstack(seq_x)
        input.append(data_in)
        output.append(seq_y)
    #print(np.shape(input))
    #print(np.shape(output))
    return np.array(input), np.array(output)


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
        input.append(seq_x)
        output.append(seq_y)
    #print(np.shape(input))
    #print(np.shape(output))
    return np.array(input), np.array(output)'''
n_steps=4
X, y= sequence(n_steps)
xgboost=xgb.Booster()
f=open('C:/Users/chang/Documents/GitHub/dataclean/dataclean/GRNN.dill', 'rb')
xgboost.load_model('C:/Users/chang/Documents/GitHub/dataclean/dataclean/xgboost_test.model')
grnn=dill.load(f)
#gru=keras.models.load_model('C:/Users/chang/Documents/GitHub/dataclean/dataclean/GRU.h5')
svr=joblib.load('C:/Users/chang/Documents/GitHub/dataclean/dataclean/model_svr.pkl')

x_train_all, x_predict, y_train_all, y_predict = train_test_split(X, y, test_size=0.10, random_state=100)
x_train, x_test, y_train, y_test = train_test_split(x_train_all, y_train_all, test_size=0.2, random_state=100)
dpredict = xgb.DMatrix(x_predict)
print(y_predict.shape)

'''a=np.array([1,2,3,4,5])
b=np.array([3,4,5,6,7])
c=b.reshape(5,1)
print(a.shape)
print(b.shape)
print(c.shape)

print(a-c)
print(mean_squared_error(a,c))
print(c.tolist())
print(a.tolist())
print(x)'''


result1=xgboost.predict(dpredict)
print(result1.shape)
print(mean_squared_error(y_predict, result1))
result1=result1.reshape(-1,1)

result2=grnn.predict(x_predict)
print(result2.shape)
print(mean_squared_error(y_predict, result2))

result3=svr.predict(x_predict)
print(result3.shape)
print(mean_squared_error(y_predict, result3))
result3=result3.reshape(-1,1)


result_predict=np.hstack([result1,result2,result3])

print(result_predict.shape)


mixed=keras.models.load_model('C:/Users/chang/Documents/GitHub/dataclean/dataclean/mixed_model.h5')
result4=mixed.predict(result_predict)
print(mean_squared_error(y_predict, result4))






'''def weight_training(argsDic):
    argsDic=argsDict_tranform(argsDic)
    print(argsDic['batch_size'])
    model=keras.models.Sequential()
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(1))
    adam = keras.optimizers.Adam(lr=argsDic['lr'], decay=argsDic['decay'])
    model.compile(optimizer=adam, loss='mean_squared_error', metrics=['mae'])
    print('start training')
    model.fit(x_train_all,y_train_all, epochs=argsDic['epochs'], batch_size=argsDic['batch_size'], validation_split=0.2)
    loss=get_tranformer_score(model)


def get_tranformer_score(tranformer):
    gru = tranformer
    prediction = gru.predict(x_predict)
    for i in prediction:
        if math.isnan(i[0]):
            print('nan number is found')
            return 10
    return mean_squared_error(y_predict, prediction)

def weight_training_best(argsDic):
    argsDic=argsDict_tranform(argsDic)
    print(argsDic['batch_size'])
    model=keras.models.Sequential()
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(1))
    adam = keras.optimizers.Adam(lr=argsDic['lr'], decay=argsDic['decay'])
    model.compile(optimizer=adam, loss='mean_squared_error', metrics=['mae'])
    print('start training')
    model.fit(x_train_all,y_train_all, epochs=argsDic['epochs'], batch_size=argsDic['batch_size'], validation_split=0.2)
    loss=get_tranformer_score(model)'''


