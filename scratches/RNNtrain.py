#import pandas as pd
import numpy as np
import csv
#import matplotlib.pyplot as plt
from math import sqrt
from tensorflow import keras
import os
from keras_layer_normalization import LayerNormalization
from loss import LossHistory
from sklearn.utils import shuffle
# model itself
#from keras.models import Sequential
#from keras.layers import LSTM
#from keras.layers import Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
history = LossHistory()
def dataReader():
    file=open("/home/xcha8737/Solar_Forecast/trainning_data/SolarPrediction.csv/SolarPrediction.csv", 'r', encoding='utf-8' )
    reader=csv.reader(file)
    features=[]
    output=[]
    for row in reader:
        feature=[]
        #feature.append(row[1])
        #feature.append(row[2])
        feature.append(float(row[3]))
        feature.append(float(row[4]))
        feature.append(float(row[5]))
        feature.append(float(row[6]))
        feature.append(float(row[7]))
        feature.append(float(row[8]))
        features.append(feature)
        #features.append(row[4:9])
        output.append(float(row[3]))
    file.close()
    sep_in=[]
    sep_out=[]
    oct_in=[]
    oct_out=[]
    Nov_in=[]
    Nov_out=[]
    Dec_in=[]
    Dec_out=[]
    i=7416
    #print(features[0])
    while(i>-1):
        sep_in.append(features[i])
        sep_out.append(output[i])
        i-=1
    i=16237
    while(i>7416):
        oct_in.append(features[i])
        oct_out.append(output[i])
        i-=1
    i=24521
    while(i>16237):
        Nov_in.append(features[i])
        Nov_out.append(output[i])
        i-=1
    i=len(features)-1
    while(i>24521):
        Dec_in.append(features[i])
        Dec_out.append(output[i])
        i-=1
    input=sep_in + oct_in + Nov_in + Dec_in
    output=sep_out+ oct_out +Nov_out+ Dec_out
    #print(len(input))
    #print(len(output))
    X=np.array(input)
    Y=np.array(output)
    #print(X[0])
    #print(Y[0])
    return X, Y

#x,y=dataReader()
def sequence( n_steps):
    X,Y=dataReader()
    x=(X-X.mean(axis=0))/X.std(axis=0)
    y=(Y-Y.mean(axis=0))/Y.std(axis=0)
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


def LSTM_training():
    n_steps=16
    X, y= sequence(n_steps)
    print(X.shape)
    print(y.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    #input, out=X[8002:8003,:],y[8002:8003]
    #train_X=np.array(input)
    #train_y=np.array(out)
    model=keras.models.Sequential()
    #model.add(keras.layers.LayerNormalization())
    model.add(keras.layers.LSTM(128, activation='relu', return_sequences=True, input_shape=(n_steps, 6),dropout=0.3, recurrent_dropout=0.3))
    model.add(keras.layers.LayerNormalization())
    model.add(keras.layers.LSTM(64, activation='relu',dropout=0.3, recurrent_dropout=0.3))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(1))
    model.compile(optimizer='adam', loss='mse')
    print('start training')
    model.fit(X_train, y_train, epochs=500, batch_size=500)
    model.save('/home/xcha8737/Solar_Forecast/trainning_data/SolarPrediction.csv/LSTM.h5')

LSTM_training()








