#import pandas as pd
import numpy as np
import csv
import math
from math import sqrt
import keras
import os
from keras_layer_normalization import LayerNormalization
from loss import LossHistory
from sklearn.utils import shuffle
from hyperopt import fmin, tpe, hp, partial, Trials, STATUS_OK, STATUS_FAIL
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, zero_one_loss, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import time
from math import sqrt
from sklearn.metrics import mean_squared_error, zero_one_loss,mean_absolute_error,r2_score

# model itself
#from keras.models import Sequential
#from keras.layers import LSTM
#from keras.layers import Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
history = LossHistory()

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
    # print(X.shape)
    # print(Y.shape)
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
        input.append(seq_x)
        output.append(seq_y)
    #print(np.shape(input))
    #print(np.shape(output))
    return np.array(input), np.array(output)
n_steps=12
X, Y= sequence(n_steps)
X=(X-X.min(axis=0))/(X.max(axis=0)-X.min(axis=0))
y=(Y-Y.min(axis=0))/(Y.max(axis=0)-Y.min(axis=0))

x_train_all, x_predict, y_train_all, y_predict = train_test_split(X, y, test_size=0.10, random_state=100)
x_train, x_test, y_train, y_test = train_test_split(x_train_all, y_train_all, test_size=0.2, random_state=100)

space = {"layer1_output": hp.randint("layer1_output", 200),
         "layer2_output": hp.randint("layer2_output", 200),
         "layer1_dropout": hp.uniform("layer1_dropout", 0, 1),
         "layer2_dropout": hp.uniform("layer2_dropout", 0, 1),
         "layer1_rdropout": hp.uniform('layer1_rdropout', 0, 1),
         "layer2_rdropout": hp.uniform('layer2_rdropout', 0, 1),
         "layer3_dropout": hp.uniform('layer3_dropout', 0, 1),
         #"optimizer": hp.choice('optimizer', ['adam', 'sgd']),
         "momentum": hp.uniform('momentum', 0,1),
         "lr": hp.uniform('lr', 1e-9, 1e-3),
         "decay": hp.uniform('decay', 1e-9, 1e-3),
         'epochs': hp.randint('epochs', 250),
         'batch_size': hp.randint('batch_size', 100)
         }

def argsDict_tranform(argsDict):
    argsDict["layer1_output"] = argsDict["layer1_output"] + 20
    argsDict['layer2_output'] = argsDict['layer2_output'] + 20
    argsDict['epochs'] = argsDict['epochs'] + 50
    argsDict['batch_size'] = argsDict['batch_size'] + 32
    return argsDict


def GRU_training(argsDic):
    argsDic=argsDict_tranform(argsDic)
    print(argsDic['batch_size'])
    model=keras.models.Sequential()
    model.add(LayerNormalization())
    model.add(keras.layers.LSTM(argsDic['layer1_output'], activation='relu', return_sequences=True, input_shape=(n_steps, 13),dropout=argsDic['layer1_dropout'], recurrent_dropout=argsDic['layer1_rdropout']))
    model.add(LayerNormalization())
    model.add(keras.layers.LSTM(argsDic['layer2_output'], activation='relu',dropout=argsDic['layer2_dropout'], recurrent_dropout=argsDic['layer2_rdropout']))
    model.add(keras.layers.Dropout(argsDic['layer3_dropout']))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(1))
    adam = keras.optimizers.Adam(lr=argsDic['lr'], decay=argsDic['decay'])
    model.compile(optimizer=adam, loss='mean_squared_error', metrics=['mae'])
    print('start training')
    model.fit(x_train_all,y_train_all, epochs=argsDic['epochs'], batch_size=argsDic['batch_size'], validation_split=0.2)
    #model.fit(x_train_all, y_train_all, epochs=1, batch_size=argsDic['batch_size'],validation_split=0.2)
    loss=get_tranformer_score(model)
    if(loss==10):
        return {'loss':loss, 'status':STATUS_FAIL}
    #model.save('/home/xcha8737/Solar_Forecast/trainning_data/SolarPrediction.csv/GRU.h5')
    else:
        return {'loss':loss, 'status':STATUS_OK}

def get_tranformer_score(tranformer):
    gru = tranformer
    prediction = gru.predict(x_predict)
    for i in prediction:
        if math.isnan(i[0]):
            print('nan number is found')
            return 10
    return mean_squared_error(y_predict, prediction)

def GRU_training_best(argsDic):
    argsDic=argsDict_tranform(argsDic)
    model=keras.models.Sequential()
    model.add(LayerNormalization())
    model.add(keras.layers.LSTM(argsDic['layer1_output'], activation='relu', return_sequences=True, input_shape=(n_steps, 13),dropout=argsDic['layer1_dropout'], recurrent_dropout=argsDic['layer1_rdropout']))
    model.add(LayerNormalization())
    model.add(keras.layers.LSTM(argsDic['layer2_output'], activation='relu',dropout=argsDic['layer2_dropout'], recurrent_dropout=argsDic['layer2_rdropout']))
    model.add(keras.layers.Dropout(argsDic['layer3_dropout']))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(1))
    adam = keras.optimizers.Adam(lr=argsDic['lr'], decay=argsDic['decay'])
    model.compile(optimizer=adam, loss='mean_squared_error', metrics=['mae'])
    print('start training')
    model.fit(x_train_all,y_train_all, epochs=argsDic['epochs'], batch_size=argsDic['batch_size'], validation_split=0.2)
    time_start=time.time()
    result=model.predict(x_predict)
    time_end=time.time()
    result = result * ((Y.max(axis=0) - Y.min(axis=0))) + Y.min(axis=0)
    y_real = y_predict * ((Y.max(axis=0) - Y.min(axis=0))) + Y.min(axis=0)
    print('totally cost', time_end - time_start)
    print("rmse is ：", sqrt(mean_squared_error(y_real, result)))
    print("mae is ：", mean_absolute_error(y_real, result))
    print('r2 is :', r2_score(y_real, result))

    '''time_start = time.time()
    result4 = model.predict(x_predict)
    time_end = time.time()
    result4 = result4.reshape(-1, 1)
    result4 = result4 * ((Y.max(axis=0) - Y.min(axis=0))) + Y.min(axis=0)
    y_test1=y_predict
    y_test1 = y_test1 * ((Y.max(axis=0) - Y.min(axis=0))) + Y.min(axis=0)
    print(mean_squared_error(y_test1, result4))
    print(mean_absolute_error(y_test1, result4))
    print(np.sqrt(mean_squared_error(y_test1, result4)))
    print(r2_score(y_test1, result4))
    print('totally cost', time_end - time_start)'''


    model.save('/home/xcha8737/Solar_Forecast/trainning_data/dataclean/dataclean/LSTM.h5')
    return {'loss':get_tranformer_score(model), 'status':STATUS_OK}



#GRU_training()
#history.loss_plot('epoch')
trials = Trials()
algo = partial(tpe.suggest, n_startup_jobs=20)
best = fmin(GRU_training, space, algo=algo, max_evals=100, pass_expr_memo_ctrl=None, trials=trials)

time_start=time.time()
MSE = GRU_training_best(best)
time_end=time.time()
print('training cost is: ', time_end-time_start)
print('best :', best)
print('best param after transform :')
print(argsDict_tranform(best))
print('\nrmse of the best gru:', np.sqrt(MSE['loss']))
print ('trials:')
# xs0 = [t['misc']['vals']['lr'][0] for t in trials.trials]
# xs1 = [t['misc']['vals']['decay'][0] for t in trials.trials]
# xs2 = [t['misc']['vals']['layer1_output'][0] for t in trials.trials]
# xs3=[t['misc']['vals']['layer2_output'][0] for t in trials.trials]
# ys=[t['result']['loss'] for t in trials.trials]
#
# plt.figure()
# plt.scatter(xs0, ys,  s=20, color='darkorange', linewidth=0.01, alpha=0.75, label='learning rate')
# plt.grid(True)
# plt.xlabel('learning rate')
# plt.ylabel('loss')
# plt.legend(loc="upper right")
# plt.show()
#
# plt.figure()
# plt.scatter(xs1, ys, s=20,  color='r',linewidth=0.01, alpha=0.75, label='decay')
# #plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
# plt.grid(True)
# plt.xlabel('decay')
# plt.ylabel('loss')
# plt.legend(loc="upper right")
# plt.show()
#
# plt.figure()
# plt.scatter(xs2, ys, s=20, color='blue', linewidth=0.01, alpha=0.75, label='layer1_output')
# #plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
# plt.grid(True)
# plt.xlabel('layer1_output')
# plt.ylabel('loss')
# plt.legend(loc="upper right")
# plt.show()
#
# plt.figure()
# plt.scatter(xs3, ys, s=20, color='g', linewidth=0.01, alpha=0.75, label='layer2_output')
# #plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
# plt.grid(True)
# plt.xlabel('layer2_output')
# plt.ylabel('loss')
# plt.legend(loc="upper right")
# plt.show()
