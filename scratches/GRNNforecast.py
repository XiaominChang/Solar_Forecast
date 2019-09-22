import numpy as np
from sklearn import datasets, preprocessing
from neupy import algorithms
import csv
import dill
import math
from math import sqrt
import keras
import os
from keras_layer_normalization import LayerNormalization
from loss import LossHistory
from sklearn.utils import shuffle
from hyperopt import fmin, tpe, hp, partial, Trials, STATUS_OK, STATUS_FAIL
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, zero_one_loss
import matplotlib.pyplot as plt
import joblib
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
n_steps=4
X, y= sequence(n_steps)
print(X.shape)
print(y.shape)
x_train_all, x_predict, y_train_all, y_predict = train_test_split(X, y, test_size=0.10, random_state=100)

space = {'std': hp.uniform('std', 0.1, 1.0)}

def GRNNtrain(argsDic):
    print('std value is:', argsDic['std'])
    model=algorithms.GRNN(std=argsDic['std'], verbose=True)
    model.train(x_train_all,y_train_all)
    loss=get_tranformer_score(model)
    if(loss==10):
        return {'loss':loss, 'status':STATUS_FAIL}
    #model.save('/home/xcha8737/Solar_Forecast/trainning_data/SolarPrediction.csv/GRU.h5')
    else:
        return {'loss':loss, 'status':STATUS_OK}



def GRNNtrain_best(argsDic):
    model=algorithms.GRNN(std=argsDic['std'], verbose=True)
    model.train(x_train_all,y_train_all)
    with open('C:/Users/chang/Documents/GitHub/dataclean/dataclean/GRNN.dill', 'wb') as f:
        dill.dump(model,f)
    return {'loss': get_tranformer_score(model), 'status': STATUS_OK}



def get_tranformer_score(tranformer):
    grnn = tranformer
    prediction = grnn.predict(x_predict)
    for i in prediction:
        if math.isnan(i[0]):
            print('nan number is found')
            return 10
    return mean_squared_error(y_predict, prediction)


trials = Trials()
algo = partial(tpe.suggest, n_startup_jobs=10)
best = fmin(GRNNtrain, space, algo=algo, max_evals=200, pass_expr_memo_ctrl=None, trials=trials)
print('best :', best)
MSE = GRNNtrain_best(best)
print('best :', best)
print('rmse of the best svr:', np.sqrt(MSE['loss']))





