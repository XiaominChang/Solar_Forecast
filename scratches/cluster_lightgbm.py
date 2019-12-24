from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import csv
from tensorflow import keras
import matplotlib.pyplot as plt
from hyperopt import fmin, tpe, hp, partial, Trials, STATUS_OK
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, zero_one_loss, r2_score
import xgboost as xgb
from sklearn.model_selection import train_test_split
import pandas as pd
import graphviz
import shap
import lightgbm as lgb
import time
from math import sqrt




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

    #x=(X-X.min(axis=0))/(X.max(axis=0)-X.min(axis=0))
    #y=(Y-Y.min(axis=0))/(Y.max(axis=0)-Y.min(axis=0))
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

#x_stan, Y= dataReader()
#n_steps=1
#X, y= sequence(n_steps)
#X,Y=dataReader()
#X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
#y = (Y - Y.min(axis=0)) / (Y.max(axis=0) - Y.min(axis=0))
#print(X.shape)
#print(y.shape)



pdata=pd.read_csv('/home/xcha8737/Downloads/cap/dataclean/labels_12.csv')
data=np.array(pdata)
X=data[:,:-1]
Y=data[:,-1]

x_train_all, x_predict, y_train_all, y_predict = train_test_split(X, Y, test_size=0.10, random_state=100)
x_train, x_test, y_train, y_test = train_test_split(x_train_all, y_train_all, test_size=0.2, random_state=100)

'''x_train_all, x_predict, y_train_all, y_predict = train_test_split(X, Y, test_size=0.10, random_state=100)

x_train, x_test, y_train, y_test = train_test_split(x_train_all, y_train_all, test_size=0.2, random_state=100)

 
x_train_all=pd.DataFrame(x_train_all,columns=["ghi", "ghi90","ghi10", "ebh", "dni", "dni10", "dni90", "dhi", "air_temp", "zenith", "azimuth", "cloud_opacity"])
x_predict=pd.DataFrame(x_predict,columns=["ghi", "ghi90","ghi10", "ebh", "dni", "dni10", "dni90", "dhi", "air_temp", "zenith", "azimuth", "cloud_opacity"])
x_train=pd.DataFrame(x_train,columns=["ghi", "ghi90","ghi10", "ebh", "dni", "dni10", "dni90", "dhi", "air_temp", "zenith", "azimuth", "cloud_opacity"])
x_test=pd.DataFrame(x_test,columns=["ghi", "ghi90","ghi10", "ebh", "dni", "dni10", "dni90", "dhi", "air_temp", "zenith","azimuth", "cloud_opacity"])'''
def LSTM_build(input, output, n_steps):
    #X, y = sequence(n_steps)
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    model=keras.models.Sequential()
    #model.add(keras.layers.LayerNormalization())
    model.add(keras.layers.LSTM(output, activation='relu', recurrent_initializer='random_uniform', bias_initializer='random_uniform', input_shape=(n_steps,13)))
    model.compile(optimizer='adam', loss='mse')
    output=model.predict(input)
    return output, model


class loss_man:
    loss_global=100000
    dtrain_global = None
    dtest_global = None
    model=None
    x_predict=None
    y_predict=None
    time_step=None
    def set_loss(self,loss):
       self.loss_global=loss
    def set_model(self,model):
       self.model=model
    def setData(self,dtrain,dtest):
        self.dtrain_global=dtrain
        self.dtest_global=dtest
    def setPredict(self, x_predict, y_predict):
        self.x_predict=x_predict
        self.y_predict=y_predict
    def setTime(self, time_step):
        self.time_step=time_step

loss_glo=loss_man()


#dtrain = xgb.DMatrix(data=x_train,label=y_train,missing=-999.0,feature_names=["ghi", "ghi90","ghi10", "ebh", "dni", "dni10", "dni90", "dhi", "air_temp", "zenith", "azimuth", "cloud_opacity"])
#dtest = xgb.DMatrix(data=x_test,label=y_test,missing=-999.0, feature_names=["ghi", "ghi90","ghi10", "ebh", "dni", "dni10", "dni90", "dhi", "air_temp", "zenith", "azimuth", "cloud_opacity"])



space = {'num_leaves': hp.randint("num_leaves", 50),
         'min_data_in_leaf': hp.randint("min_data_in_leaf", 200),
         'max_depth': hp.randint("max_depth", 10),
         'learning_rate':  hp.uniform('learning_rate', 0, 0.5),
         'feature_fraction': hp.uniform("feature_fraction", 0,1),
         #'bagging_fraction': hp.uniform("bagging_fraction", 0,1),
         'max_bin':hp.randint('max_bin', 200),
         'num_boost_round': hp.randint('num_boost_round', 200),
         #'bagging_freq': hp.randint('bagging_freq', 10),
         'min_data_in_bin':hp.randint('min_data_in_bin', 100),
         'lambda_l2': hp.uniform("lambda_l2", 0,1),
         'bin_construct_sample_cnt': hp.randint('bin_construct_sample_cnt', 100000),
         "time_step": hp.randint("time_step", 14),
         "output": hp.randint("output", 200)
}
def argsDict_tranform(argsDict, isPrint=False):
    argsDict["max_depth"] = argsDict["max_depth"] + 1
    argsDict['min_data_in_leaf'] = argsDict['min_data_in_leaf'] + 50
    argsDict["num_leaves"] = argsDict["num_leaves"] + 10
    #argsDict["learning_rate"] = argsDict["learning_rate"] * 0.02 + 0.05
    argsDict['max_bin'] = argsDict['max_bin'] + 20
    argsDict['num_boost_round'] = argsDict['num_boost_round'] + 100
    #argsDict['bagging_freq'] = argsDict['bagging_freq'] + 1
    argsDict['min_data_in_bin'] = argsDict['min_data_in_bin'] + 20
    argsDict['bin_construct_sample_cnt']=argsDict['bin_construct_sample_cnt']+20000
    argsDict['time_step']=argsDict['time_step']+1
    argsDict['output']=argsDict["output"]+32
    return argsDict


def lgb_factory(argsDict):
    argsDict = argsDict_tranform(argsDict)
    #print('time_step is',argsDict['time_step'])
    #X, y = sequence(argsDict['time_step'])
    #X, y = sequence(1)
    #X,model = LSTM_build(X,argsDict['output'],argsDict['time_step'])
    #x_train_all, x_predict, y_train_all, y_predict = train_test_split(X, y, test_size=0.10, random_state=100)
    #x_train, x_test, y_train, y_test = train_test_split(x_train_all, y_train_all, test_size=0.2, random_state=100)
    #argsDict = argsDict_tranform(argsDict)
    dtrain = lgb.Dataset(x_train, y_train)
    dtest = lgb.Dataset(x_test, y_test)
    params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': {'l1', 'l2'},
        'num_leaves': argsDict['num_leaves'],
        # 'num_leaves': 2,
        'min_data_in_leaf': argsDict['min_data_in_leaf'],
        'max_depth': argsDict['max_depth'],
        'learning_rate': argsDict['learning_rate'],
        # 'feature_fraction':argsDict['feature_fraction'],
        # 'bagging_fraction': argsDict['bagging_fraction'],
        # 'bagging_freq': argsDict['bagging_freq'],
        'max_bin': argsDict['max_bin'],
        # 'num_boost_round':argsDict['num_boost_rount'] ,
        'min_data_in_bin': argsDict['min_data_in_bin'],
        'lambda_l2': argsDict['lambda_l2'],
        'verbose': 3,
        'is_provide_training_metric': True,
        'bin_construct_sample_cnt':argsDict['bin_construct_sample_cnt']
    }
    # params = {
    #     'boosting_type': 'gbdt',
    #     'objective': 'regression',
    #     'metric': {'l1', 'l2'},
    #     'num_leaves': 42,
    #     # 'num_leaves': 2,
    #     'min_data_in_leaf': 261,
    #     'max_depth': 7,
    #     'learning_rate': 0.0923197313825187,
    #     # 'feature_fraction':argsDict['feature_fraction'],
    #     # 'bagging_fraction': argsDict['bagging_fraction'],
    #     # 'bagging_freq': argsDict['bagging_freq'],
    #     'max_bin': 149,
    #     # 'num_boost_round':argsDict['num_boost_rount'] ,
    #     'min_data_in_bin': 128,
    #     'lambda_l2': 0.5019531135206157,
    #     'verbose': -1,
    #     'is_provide_training_metric': True,
    #     'bin_construct_sample_cnt': 114852
    # }
    print("max bin is", argsDict['max_bin'])
    gbm = lgb.train(params, dtrain, num_boost_round=argsDict['num_boost_round'] , valid_sets=dtest, early_stopping_rounds=10)
    loss=get_tranformer_score(gbm, x_predict, y_predict)
    if(loss<loss_glo.loss_global):
        loss_glo.set_loss(loss)
        #loss_glo.set_model(model)
        loss_glo.setData(dtrain,dtest)
        #loss_glo.setPredict(x_predict,y_predict)
        #loss_glo.setTime(argsDict['time_step'])
    return {'loss': loss, 'status': STATUS_OK}

def get_tranformer_score(tranformer, x_predict, y_predict):
    xrf = tranformer
    #dpredict = lgb.Dataset(x_predict)
    prediction = xrf.predict(x_predict)
    #prediction = prediction * ((Y.max(axis=0) - Y.min(axis=0))) + Y.min(axis=0)
    #y_predict = y_predict * ((Y.max(axis=0) - Y.min(axis=0))) + Y.min(axis=0)
    return np.sqrt(mean_squared_error(y_predict, prediction))

def lgbbest_train(argsDict):
    argsDict = argsDict_tranform(argsDict)
    dtrain = loss_glo.dtrain_global
    dtest = loss_glo.dtest_global
    params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': {'l1', 'l2'},
        'num_leaves': argsDict['num_leaves'],
        # 'num_leaves': 2,
        'min_data_in_leaf': argsDict['min_data_in_leaf'],
        'max_depth': argsDict['max_depth'],
        'learning_rate': argsDict['learning_rate'],
        # 'feature_fraction':argsDict['feature_fraction'],
        # 'bagging_fraction': argsDict['bagging_fraction'],
        # 'bagging_freq': argsDict['bagging_freq'],
        'max_bin': argsDict['max_bin'],
        # 'num_boost_round':argsDict['num_boost_rount'] ,
        'min_data_in_bin': argsDict['min_data_in_bin'],
        'lambda_l2': argsDict['lambda_l2'],
        'verbose': -1,
        'is_provide_training_metric': True,
        'bin_construct_sample_cnt': argsDict['bin_construct_sample_cnt']
    }
    # params = {
    #     'boosting_type': 'gbdt',
    #     'objective': 'regression',
    #     'metric': {'l1', 'l2'},
    #     'num_leaves': 42,
    #     # 'num_leaves': 2,
    #     'min_data_in_leaf': 261,
    #     'max_depth': 7,
    #     'learning_rate': 0.0923197313825187,
    #     # 'feature_fraction':argsDict['feature_fraction'],
    #     # 'bagging_fraction': argsDict['bagging_fraction'],
    #     # 'bagging_freq': argsDict['bagging_freq'],
    #     'max_bin': 149,
    #     # 'num_boost_round':argsDict['num_boost_rount'] ,
    #     'min_data_in_bin': 128,
    #     'lambda_l2': 0.5019531135206157,
    #     'verbose': -1,
    #     'is_provide_training_metric': True,
    #     'bin_construct_sample_cnt': 114852
    # }

    gbm = lgb.train(params, dtrain, num_boost_round=argsDict['num_boost_round'], valid_sets=dtest, early_stopping_rounds=10)
    loss=get_tranformer_score(gbm,x_predict,y_predict)
    gbm.save_model('/home/xcha8737/Solar_Forecast/trainning_data/dataclean/dataclean/GBMmodel_test00022.txt')
    time_start=time.time()
    result=gbm.predict(x_predict)
    time_end=time.time()
    print('totally cost', time_end - time_start)
    print("rmse is ：", sqrt(mean_squared_error(y_predict, result)))
    print("mae is ：", mean_absolute_error(y_predict, result))
    print('r2 is :', r2_score(y_predict, result))
    #dtrain.save_binary("/home/xcha8737/Solar_Forecast/trainning_data/dataclean/dataclean/train001.bin")
    #dtest.save_binary("/home/xcha8737/Solar_Forecast/trainning_data/dataclean/dataclean/test001.bin")
    return {'loss': loss, 'status': STATUS_OK}
    #xrf.save_model('/home/xcha8737/Solar_Forecast/trainning_data/dataclean/dataclean/xgboost.model')


# def lgbbest_train1(argsDict):
# #def lgbbest_train():
#     argsDict = argsDict_tranform(argsDict)
#     dtrain1=lgb.Dataset('/home/xcha8737/Solar_Forecast/trainning_data/dataclean/dataclean/train.bin')
#     dtest1=lgb.Dataset('/home/xcha8737/Solar_Forecast/trainning_data/dataclean/dataclean/test.bin')
#     print(dtrain1)
#     params = {
#         'boosting_type': 'gbdt',
#         'objective': 'regression',
#         'metric': {'l1', 'l2'},
#         'num_leaves': argsDict['num_leaves'],
#         # 'num_leaves': 2,
#         'min_data_in_leaf': argsDict['min_data_in_leaf'],
#         'max_depth': argsDict['max_depth'],
#         'learning_rate': argsDict['learning_rate'],
#         # 'feature_fraction':argsDict['feature_fraction'],
#         # 'bagging_fraction': argsDict['bagging_fraction'],
#         # 'bagging_freq': argsDict['bagging_freq'],
#         'max_bin': argsDict['max_bin'],
#         # 'num_boost_round':argsDict['num_boost_rount'] ,
#         'min_data_in_bin': argsDict['min_data_in_bin'],
#         'lambda_l2': argsDict['lambda_l2'],
#         'verbose': -1,
#         'is_provide_training_metric': True
#     }
#
#     #gbm = lgb.train(params, dtrain1, num_boost_round=argsDict['num_boost_round'] , valid_sets=dtest1, early_stopping_rounds=10)
#     gbm = lgb.train(params, dtrain1, num_boost_round=argsDict['num_boost_round'], valid_sets=dtest1, early_stopping_rounds=10)
#     loss=get_tranformer_score(gbm)
#     print('loss is :', loss)
#     print('starting test')
#
#     gbm.save_model('/home/xcha8737/Solar_Forecast/trainning_data/dataclean/dataclean/GBMmodel_test1101.txt')
#     return {'loss': loss, 'status': STATUS_OK}
#best=lgbbest_train()
#print('\nrmse of the best lightgbm:', np.sqrt(best['loss']))

trials = Trials()
algo = partial(tpe.suggest, n_startup_jobs=20)
best = fmin(lgb_factory, space, algo=algo, max_evals=3000, pass_expr_memo_ctrl=None, trials=trials)
time_start=time.time()
RMSE= lgbbest_train(best)
time_end=time.time()
print('time cost is :', time_end-time_start)
print('best :', best)
#print('best param after transform :')
#print(argsDict_tranform(best, isPrint=True))
print('\nrmse of the best lightgbm:', RMSE['loss'])
# print('time_step is:', loss_glo.time_step)
# print([t['misc']['vals']['time_step'][0]+1 for t in trials.trials])
# xs0 = [t['misc']['vals']['time_step'][0]+1 for t in trials.trials]
# ys=[t['result']['loss'] for t in trials.trials]
# xs0=np.array(xs0)
# ys=np.array(ys)
# xs0=xs0.reshape([-1,1])
# ys=ys.reshape([-1,1])
# data=np.hstack((xs0,ys))
# pandata=pd.DataFrame(data,columns=['x','y'])
# pandata.to_csv('/home/xcha8737/Downloads/cap/dataclean/test_timestep.csv')
# plt.figure()
# plt.scatter(xs0, ys,  s=20, color='darkorange', linewidth=0.01, alpha=0.75, label='time_step')
# plt.grid(True)
# plt.xlabel('time_step')
# plt.ylabel('loss')
# plt.legend(loc="upper right")
# plt.show()

'''MSE= lgbbest_train(best)
MSE1= lgbbest_train1(best)
MSE2= lgbbest_train1(best)
MSE3= lgbbest_train1(best)
MSE4= lgbbest_train1(best)
MSE5= lgbbest_train1(best)
print('\nrmse of the best lightgbm:', np.sqrt(MSE['loss']))
print('\nrmse of the best lightgbm:', np.sqrt(MSE1['loss']))
print('\nrmse of the best lightgbm:', np.sqrt(MSE2['loss']))
print('\nrmse of the best lightgbm:', np.sqrt(MSE3['loss']))
print('\nrmse of the best lightgbm:', np.sqrt(MSE4['loss']))
print('\nrmse of the best lightgbm:', np.sqrt(MSE5['loss']))'''

'''MSE1= lgbbest_train(best)
MSE2= lgbbest_train(best)
MSE3= lgbbest_train(best)
MSE4= lgbbest_train(best)
MSE5= lgbbest_train(best)
MSE6= lgbbest_train(best)
MSE7= lgbbest_train(best)
MSE8= lgbbest_train(best)
MSE9= lgbbest_train(best)
MSE10= lgbbest_train(best)

print('\nrmse of the best lightgbm:', np.sqrt(MSE1['loss']))
print('\nrmse of the best lightgbm:', np.sqrt(MSE2['loss']))
print('\nrmse of the best lightgbm:', np.sqrt(MSE3['loss']))
print('\nrmse of the best lightgbm:', np.sqrt(MSE4['loss']))
print('\nrmse of the best lightgbm:', np.sqrt(MSE5['loss']))
print('\nrmse of the best lightgbm:', np.sqrt(MSE6['loss']))
print('\nrmse of the best lightgbm:', np.sqrt(MSE7['loss']))
print('\nrmse of the best lightgbm:', np.sqrt(MSE8['loss']))
print('\nrmse of the best lightgbm:', np.sqrt(MSE9['loss']))
print('\nrmse of the best lightgbm:', np.sqrt(MSE10['loss']))'''


#MSE = lgbbest_train()
#print('\nrmse of the best lightgbm:', np.sqrt(MSE['loss']))
'''  params = {
    'boosting_type': 'gdbt',
    'objective': 'regression',
    'metric': {'l1', 'l2'},
    'num_leaves': 62,
    # 'num_leaves': 2,
    'min_data_in_leaf': 117,
    'max_depth': 8,
    'learning_rate': 0.016160222735729687,
    # 'feature_fraction':argsDict['feature_fraction'],
    # 'bagging_fraction': argsDict['bagging_fraction'],
    # 'bagging_freq': argsDict['bagging_freq'],
    'max_bin': 228,
    # 'num_boost_round':argsDict['num_boost_rount'] ,
    'min_data_in_bin': 101,
    'lambda_l2': 0.9031062690582833,
    'verbose': 1,
    'is_provide_training_metric': True
}'''


