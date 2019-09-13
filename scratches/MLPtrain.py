import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
from math import sqrt
from tensorflow import keras
import os
# model itself
from loss import LossHistory
from sklearn.utils import shuffle
#from keras.models import Sequential
#from keras.layers import LSTM
#from keras.layers import Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error

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

def mlp_training():
    n_steps=16
    X, y= sequence(n_steps)
    train_X, train_y=shuffle(X,y, random_state=0)
    #train_X, train_y=X[:-1000,:],y[:-1000]
    #test_X, test_y=X[-1000:,:], y[-1000:]
    #input, out=X[8002:8003,:],y[8002:8003]
    #train_X=np.array(input)
    #train_y=np.array(out)
    print(np.shape(train_X))
    print(np.shape(train_y))
    #model=keras.models.load_model('/home/xcha8737/Solar_Forecast/trainning_data/SolarPrediction.csv/mlp01.h5')
    #model.add(keras.layers.LSTM(6, activation='relu',return_sequences=True, input_shape=(n_steps, 6)))
    #model.add(keras.layers.Dropout(0.2))
    #model.add(keras.layers.LSTM(32, activation='relu',kernel_initializer='glorot_uniform',return_sequences=False, input_shape=(n_steps, 6)))
    #model.add(keras.layers.Dropout(0.2))
    #model.add(keras.layers.Flatten())
    #model.add(keras.layers.LSTM(64, activation='tanh'))
    train_X=train_X.reshape([32670,96])
    model=keras.models.Sequential()
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(400,activation='sigmoid', kernel_initializer='glorot_uniform'))
    model.add(keras.layers.Dropout(0.4))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(300, activation='sigmoid', kernel_initializer='glorot_uniform'))
    model.add(keras.layers.Dropout(0.4))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(200, activation='sigmoid', kernel_initializer='glorot_uniform'))
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(1,activation='relu',kernel_initializer='glorot_uniform'))
    #rms = keras.optimizers.RMSprop(lr=0.01, rho=0.9, epsilon=1e-08, decay=0.01)
    #model.load_weights('C:/Users/chang/Documents/GitHub/Solar_Forecast/trainning_data/LSTM.h5')
    #sgd=keras.optimizers.SGD(lr=1e-10, decay=0.1, momentum=0.8, nesterov=True)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    print('start training')
    model.fit(train_X,train_y, epochs=65, batch_size=32,callbacks=[history], validation_split=0.3)
    model.save('/home/xcha8737/Solar_Forecast/trainning_data/SolarPrediction.csv/mlp02.h5')

mlp_training()

history.loss_plot('epoch')







