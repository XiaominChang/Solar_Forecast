import numpy as np
from sklearn import datasets, preprocessing
from neupy import algorithms
import csv
import dill
import math
from math import sqrt
#import keras
import os
#from keras_layer_normalization import LayerNormalization
#from loss import LossHistory
from sklearn.utils import shuffle
from hyperopt import fmin, tpe, hp, partial, Trials, STATUS_OK, STATUS_FAIL
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
import joblib
import pandas as pd
import shap
import time
from sklearn.metrics import mean_squared_error, zero_one_loss,mean_absolute_error,r2_score


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
        output.append(seq_y)
    #print(np.shape(input))
    #print(np.shape(output))
    return np.array(input,dtype=np.float32), np.array(output,dtype=np.float32)


n_steps=8
X, Y= sequence(n_steps)
X=(X-X.min(axis=0))/(X.max(axis=0)-X.min(axis=0))
y=(Y-Y.min(axis=0))/(Y.max(axis=0)-Y.min(axis=0))
'''X,Y=dataReader()
X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
y = (Y - Y.min(axis=0)) / (Y.max(axis=0) - Y.min(axis=0))
print(X.shape)
print(y.shape)'''
# pdata=pd.read_csv('/home/xcha8737/Downloads/cap/dataclean/labels_0.csv')
# data=np.array(pdata, dtype=np.float32)
# X=data[:,:-1]
# Y=data[:,-1]
#
# x_train_all, x_predict, y_train_all, y_predict = train_test_split(X, Y, test_size=0.10, random_state=100)
# x_train, x_test, y_train, y_test = train_test_split(x_train_all, y_train_all, test_size=0.2, random_state=100)


x_train_all, x_predict, y_train_all, y_predict = train_test_split(X, y, test_size=0.10, random_state=100)

x_train, x_test, y_train, y_test = train_test_split(x_train_all, y_train_all, test_size=0.2, random_state=100)

# x_train_all=pd.DataFrame(x_train_all,columns=["ghi", "ghi90","ghi10", "ebh", "dni", "dni10", "dni90", "dhi", "air_temp", "zenith", "azimuth", "cloud_opacity"])
# x_predict=pd.DataFrame(x_predict,columns=["ghi", "ghi90","ghi10", "ebh", "dni", "dni10", "dni90", "dhi", "air_temp", "zenith", "azimuth", "cloud_opacity"])
# x_train=pd.DataFrame(x_train,columns=["ghi", "ghi90","ghi10", "ebh", "dni", "dni10", "dni90", "dhi", "air_temp", "zenith", "azimuth", "cloud_opacity"])
# x_test=pd.DataFrame(x_test,columns=["ghi", "ghi90","ghi10", "ebh", "dni", "dni10", "dni90", "dhi", "air_temp", "zenith","azimuth", "cloud_opacity"])
# #
# data=pd.read_csv("/home/xcha8737/Desktop/test_data/all_data.csv")
# #X=data[["airtemp", "humidity", "insolation", "windspeed", "winddirection"]]
# X=data[["airtemp", "humidity", "insolation"]]
# Y=data['power (W)']



# x_train_all, x_predict, y_train_all, y_predict = train_test_split(X, Y, test_size=0.90, random_state=100)
#
# x_train, x_test, y_train, y_test = train_test_split(x_train_all, y_train_all, test_size=0.90, random_state=100)

# pdata=pd.read_csv('/home/xcha8737/Downloads/cap/dataclean/grnnlabels_4.csv')
# data=np.array(pdata)
# X=data[:,:-1]
# Y=data[:,-1]
# X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
# y = (Y - Y.min(axis=0)) / (Y.max(axis=0) - Y.min(axis=0))
# x_train_all, x_predict, y_train_all, y_predict = train_test_split(X, y, test_size=0.10, random_state=100)
# x_train, x_test, y_train, y_test = train_test_split(x_train_all, y_train_all, test_size=0.2, random_state=100)

space = {'std': hp.uniform('std', 0.1, 1.0),
         'time_step':hp.randint('time_step',13)}


def argsDict_tranform(argsDict):
    argsDict['time_step']=argsDict['time_step']+1
    return argsDict

def GRNNtrain(argsDic):
    argsDic=argsDict_tranform(argsDic)
    # n_steps = argsDic['time_step']
    # X, Y = sequence(n_steps)
    # X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    # y = (Y - Y.min(axis=0)) / (Y.max(axis=0) - Y.min(axis=0))
    # x_train_all, x_predict, y_train_all, y_predict = train_test_split(X, y, test_size=0.10, random_state=100)
    # x_train, x_test, y_train, y_test = train_test_split(x_train_all, y_train_all, test_size=0.2, random_state=100)
    print('std value is:', argsDic['std'])
    model=algorithms.GRNN(std=argsDic['std'], verbose=True)
    # model=algorithms.GRNN(std=0.503, verbose=True)
    model.train(x_train_all,y_train_all)
    loss=get_tranformer_score(model, x_predict, y_predict, Y)
    if(loss==10):
        return {'loss':loss, 'status':STATUS_FAIL}
    #model.save('/home/xcha8737/Solar_Forecast/trainning_data/SolarPrediction.csv/GRU.h5')
    else:
        return {'loss':loss, 'status':STATUS_OK}



def GRNNtrain_best(argsDic):
    # argsDic=argsDict_tranform(argsDic)
    # n_steps = argsDic['time_step']
    # X, Y = sequence(n_steps)
    # X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    # y = (Y - Y.min(axis=0)) / (Y.max(axis=0) - Y.min(axis=0))
    # x_train_all, x_predict, y_train_all, y_predict = train_test_split(X, y, test_size=0.10, random_state=100)
    # x_train, x_test, y_train, y_test = train_test_split(x_train_all, y_train_all, test_size=0.2, random_state=100)
    model=algorithms.GRNN(std=argsDic['std'], verbose=True)
    #model=algorithms.GRNN(std=0.503, verbose=True)
    model.train(x_train_all,y_train_all)

    #shap_value = shap.KernelExplainer(model.predict(), x_train_all).shap_values(x_train_all)
    #shap.summary_plot(shap_value, x_train_all, plot_type="bar")
    with open('/home/xcha8737/Solar_Forecast/trainning_data/dataclean/dataclean/grnn_test.dill', 'wb') as f:
        dill.dump(model,f)
    time_start=time.time()
    result=model.predict(x_predict)
    time_end=time.time()
    result = result * ((Y.max(axis=0) - Y.min(axis=0))) + Y.min(axis=0)
    y_real = y_predict * ((Y.max(axis=0) - Y.min(axis=0))) + Y.min(axis=0)
    print('totally cost', time_end - time_start)
    print("rmse is ：", sqrt(mean_squared_error(y_real, result)))
    print("mae is ：", mean_absolute_error(y_real, result))
    print('r2 is :', r2_score(y_real, result))

    return {'loss': get_tranformer_score(model,x_predict,y_predict,Y), 'status': STATUS_OK}



def get_tranformer_score(tranformer,x_predict,y_predict,Y):
    grnn = tranformer
    prediction = grnn.predict(x_predict)
    prediction = prediction * ((Y.max(axis=0) - Y.min(axis=0))) + Y.min(axis=0)
    y_predict1 = y_predict * ((Y.max(axis=0) - Y.min(axis=0))) + Y.min(axis=0)
    for i in prediction:
        if math.isnan(i[0]):
            print('nan number is found')
            return 10
    return np.sqrt(mean_squared_error(y_predict1, prediction))

trials = Trials()
algo = partial(tpe.suggest, n_startup_jobs=20)
best = fmin(GRNNtrain, space, algo=algo, max_evals=500, pass_expr_memo_ctrl=None, trials=trials)
print('best :', best)
time_start=time.time()
MSE = GRNNtrain_best(best)
time_end=time.time()
print('training cost is: ', time_end-time_start)
print('best :', best)
print('rmse of the best svr:', MSE['loss'])


# xs0 = [t['misc']['vals']['time_step'][0] for t in trials.trials]
# ys=[t['result']['loss'] for t in trials.trials]
# xs0=np.array(xs0)
# ys=np.array(ys)
# xs0=xs0.reshape([-1,1])
# ys=ys.reshape([-1,1])
# data=np.hstack((xs0,ys))
# pandata=pd.DataFrame(data,columns=['x','y'])
# #pandata.to_csv('/home/xcha8737/Downloads/cap/dataclean/test_timestep_grnn.csv')
# plt.figure()
# # plt.plot(xs0, ys, color='blue',   label='time_step')
# plt.scatter(xs0, ys,  s=20, color='blue',  label='time_step')
# #plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
# plt.grid(True)
# plt.xlabel('time_step')
# plt.ylabel('loss')
# plt.legend(loc="upper right")
# plt.show()

