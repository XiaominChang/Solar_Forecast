#import pandas as pd
import numpy as np
import csv
#import matplotlib.pyplot as plt
from math import sqrt
import keras
#from tensorflow import keras
import os
from keras_layer_normalization import LayerNormalization
from loss import LossHistory
import time
from sklearn.utils import shuffle
# model itself
#from keras.models import Sequential
#from keras.layers import LSTM
#from keras.layers import Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from hyperopt import fmin, tpe, hp, partial, Trials, STATUS_OK
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, zero_one_loss,r2_score
import xgboost as xgb
from sklearn.model_selection import train_test_split


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
history = LossHistory()

# def LSTM_build():
#
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
#     #input, out=X[8002:8003,:],y[8002:8003]
#     #train_X=np.array(input)
#     #train_y=np.array(out)
#     model=keras.models.Sequential()
#     #model.add(keras.layers.LayerNormalization())
#     model.add(keras.layers.LSTM(128, activation='relu', return_sequences=True, input_shape=(n_steps, 6),dropout=0.3, recurrent_dropout=0.3))
#     model.add(keras.layers.LayerNormalization())
#     model.add(keras.layers.LSTM(64, activation='relu',dropout=0.3, recurrent_dropout=0.3))
#     model.add(keras.layers.BatchNormalization())
#     model.add(keras.layers.Dense(1))
#     model.compile(optimizer='adam', loss='mse')
#     print('start training')
#     model.fit(X_train, y_train, epochs=500, batch_size=500)
#     model.save('/home/xcha8737/Solar_Forecast/trainning_data/SolarPrediction.csv/LSTM.h5')


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

x_stan, Y= dataReader()



def LSTM_build(input, output, output1, n_steps):
    #X, y = sequence(n_steps)
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    model=keras.models.Sequential()
    model.add(LayerNormalization())
    #model.add(keras.layers.LayerNormalization())
    model.add(keras.layers.LSTM(output, activation='tanh', recurrent_activation='sigmoid', recurrent_initializer='orthogonal', bias_initializer='random_uniform', input_shape=(n_steps,13)))
    model.add(keras.layers.Dense(output1, kernel_initializer=keras.initializers.random_normal(stddev=0.01)))
    model.compile(optimizer='adam', loss='mse')
    output=model.predict(input)
    return output, model

class loss_man:
    loss_global=100000
    model=None
    def set_loss(self,loss):
       self.loss_global=loss

    def setModel(self,model):
        self.model=model


loss_glo=loss_man()


#evallist = [(dtrain, 'train'),(dtest, 'eval')]

space = {"max_depth": hp.randint("max_depth", 10),
         "n_estimators": hp.randint("n_estimators", 200),
         'learning_rate': hp.uniform('learning_rate', 1e-3, 5e-1),
         "min_split_loss": hp.uniform("min_split_loss", 0, 1),
         "subsample": hp.uniform("subsample", 0,1),
         "colsample_bytree": hp.uniform("colsample_bytree", 0, 1),
         "min_child_weight": hp.randint("min_child_weight", 10),
         "objective": 'reg:squarederror',
         "booster" : 'gbtree',
         "lambda": hp.uniform('lambda', 0, 1),
         "time_step": hp.randint("time_step", 10),
         "output": hp.randint("output", 200),
         "output1": hp.randint("output1", 100)
         }
def argsDict_tranform(argsDict, isPrint=False):
    argsDict["max_depth"] = argsDict["max_depth"] + 5
    argsDict['n_estimators'] = argsDict['n_estimators'] + 100
    argsDict["learning_rate"] = argsDict["learning_rate"] * 0.02 + 0.05
    argsDict["min_child_weight"] = argsDict["min_child_weight"] + 5
    argsDict['time_step']=argsDict['time_step']+1
    argsDict['output']=argsDict["output"]+10
    argsDict['output1']=argsDict["output1"]+10
    return argsDict

def xgboost_factory(argsDict):
    argsDict = argsDict_tranform(argsDict)
    X, y = sequence(argsDict['time_step'])
    X,model = LSTM_build(X,argsDict['output'],argsDict['output1'],argsDict['time_step'])
    x_train_all, x_predict, y_train_all, y_predict = train_test_split(X, y, test_size=0.10, random_state=100)
    x_train, x_test, y_train, y_test = train_test_split(x_train_all, y_train_all, test_size=0.2, random_state=100)

    dtrain = xgb.DMatrix(data=x_train, label=y_train, missing=-999.0)
    dtest = xgb.DMatrix(data=x_test, label=y_test, missing=-999.0)
    evallist = [(dtrain, 'train'), (dtest, 'eval')]

    params = {'nthread': 4,
              'max_depth': argsDict['max_depth'],
              'n_estimators': argsDict['n_estimators'],
              'eta': argsDict['learning_rate'],
              'subsample': argsDict['subsample'],
              'min_child_weight': argsDict['min_child_weight'],
              'objective': 'reg:squarederror',
              'verbosity': 0,
              'gamma': argsDict['min_split_loss'],
              'colsample_bytree': argsDict['colsample_bytree'],
              'alpha': 0,
              'lambda': argsDict['lambda'],
              'scale_pos_weight': 0,
              'seed': 100,
              'missing': -999,
              "booster" : 'gbtree'
              }
    params['eval_metric'] = ['rmse']

    xrf = xgb.train(params, dtrain, params['n_estimators'], evallist, early_stopping_rounds=10)
    loss=get_tranformer_score(xrf, x_predict, y_predict)
    if(loss<loss_glo.loss_global):
        loss_glo.set_loss(loss)
        loss_glo.setModel(model)
    return {'loss': loss, 'status': STATUS_OK}

def get_tranformer_score(tranformer, x_predict, y_predict):
    xrf = tranformer
    dpredict = xgb.DMatrix(x_predict)
    prediction = xrf.predict(dpredict, ntree_limit=xrf.best_ntree_limit)
    prediction = prediction * ((Y.max(axis=0) - Y.min(axis=0))) + Y.min(axis=0)
    y_predict = y_predict * ((Y.max(axis=0) - Y.min(axis=0))) + Y.min(axis=0)
    return np.sqrt(mean_squared_error(y_predict, prediction))


def xgbest_train(argsDict):
    argsDict = argsDict_tranform(argsDict)
    X, y = sequence(argsDict['time_step'])
    #X = LSTM_build(X,argsDict['output'],argsDict['time_step'])
    X=loss_glo.model.predict(X)
    x_train_all, x_predict, y_train_all, y_predict = train_test_split(X, y, test_size=0.10, random_state=100)
    x_train, x_test, y_train, y_test = train_test_split(x_train_all, y_train_all, test_size=0.2, random_state=100)

    dtrain = xgb.DMatrix(data=x_train, label=y_train, missing=-999.0)
    dtest = xgb.DMatrix(data=x_test, label=y_test, missing=-999.0)
    evallist = [(dtrain, 'train'), (dtest, 'eval')]
    params = {'nthread': 4,
              'max_depth': argsDict['max_depth'],
              'n_estimators': argsDict['n_estimators'],
              'eta': argsDict['learning_rate'],
              'subsample': argsDict['subsample'],
              'min_child_weight': argsDict['min_child_weight'],
              'objective': 'reg:squarederror',
              'verbosity': 0,
              'gamma': argsDict['min_split_loss'],
              'colsample_bytree': argsDict['colsample_bytree'],
              'alpha': 0,
              'lambda': argsDict['lambda'],
              'scale_pos_weight': 0,
              'seed': 100,
              'missing': -999,
              "booster": 'gbtree'
              }
    params['eval_metric'] = ['rmse']

    xrf = xgb.train(params, dtrain, params['n_estimators'], evallist, early_stopping_rounds=10)
    loss=get_tranformer_score(xrf, x_predict, y_predict)
    dpredict = xgb.DMatrix(x_predict)
    time_start = time.time()
    result1 = xrf.predict(dpredict)
    time_end = time.time()
    result1 = result1 * ((Y.max(axis=0) - Y.min(axis=0))) + Y.min(axis=0)
    y_predict = y_predict * ((Y.max(axis=0) - Y.min(axis=0))) + Y.min(axis=0)
    print(mean_squared_error(y_predict, result1))
    print(mean_absolute_error(y_predict, result1))
    print(np.sqrt(mean_squared_error(y_predict, result1)))
    print(r2_score(y_predict, result1))
    print('totally cost', time_end - time_start)
    xrf.save_model('/home/xcha8737/Solar_Forecast/trainning_data/dataclean/dataclean/xgboost.model')
    '''print(xrf.feature_names)
    xgb.plot_importance(xrf)
    plt.show()
    shap_value=shap.TreeExplainer(xrf).shap_values(x_train_all)
    shap.summary_plot(shap_value, x_train_all, plot_type="bar")
    #fig=plt.gcf()
    plt.show()'''

    return {'loss': loss, 'status': STATUS_OK}


trials = Trials()
algo = partial(tpe.suggest, n_startup_jobs=20)
best = fmin(xgboost_factory, space, algo=algo, max_evals=1000, pass_expr_memo_ctrl=None, trials=trials)
MSE = xgbest_train(best)
print('best :', best)
print('best param after transform :')
print(argsDict_tranform(best,isPrint=True))
print('\nrmse of the best xgboost:', MSE['loss'])
#print('\nrmse of the best xgboost:', np.sqrt(MSE1['loss']))
#print('\nrmse of the best xgboost:', np.sqrt(MSE2['loss']))
#print('\nrmse of the best xgboost:', np.sqrt(MSE3['loss']))
#print('\nrmse of the best xgboost:', np.sqrt(MSE4['loss']))
# xs0 = [t['misc']['vals']['learning_rate'][0] for t in trials.trials]
# xs1 = [t['misc']['vals']['n_estimators'][0] for t in trials.trials]
# xs2 = [t['misc']['vals']['max_depth'][0] for t in trials.trials]
# xs3=[t['misc']['vals']['min_child_weight'][0] for t in trials.trials]
# ys=[t['result']['loss'] for t in trials.trials]
# best=argsDict_tranform(best)
# X, y = sequence(best['time_step'])
# X = LSTM_build(X, best['output'], best['time_step'])
# x_train_all, x_predict, y_train_all, y_predict = train_test_split(X, y, test_size=0.10, random_state=100)
# x_train, x_test, y_train, y_test = train_test_split(x_train_all, y_train_all, test_size=0.2, random_state=100)

# X, y = sequence(9)
# X = LSTM_build(X, 171, 9)
# x_train_all, x_predict, y_train_all, y_predict = train_test_split(X, y, test_size=0.10, random_state=100)
# x_train, x_test, y_train, y_test = train_test_split(x_train_all, y_train_all, test_size=0.2, random_state=100)
#
# xgboost=xgb.Booster()
# xgboost.load_model('/home/xcha8737/Solar_Forecast/trainning_data/dataclean/dataclean/xgboost.model')
# dpredict = xgb.DMatrix(x_predict)
# time_start=time.time()
# #result1=lgbm.predict(x_predict)
# result1=xgboost.predict(dpredict)
# time_end=time.time()
# result1=result1*((Y.max(axis=0) - Y.min(axis=0)))+Y.min(axis=0)
# y_predict=y_predict*((Y.max(axis=0) - Y.min(axis=0)))+Y.min(axis=0)
# print(mean_squared_error(y_predict, result1))
# print(mean_absolute_error(y_predict, result1))
# print(np.sqrt(mean_squared_error(y_predict, result1)))
# print(r2_score(y_predict,result1))
# print('totally cost',time_end-time_start)
