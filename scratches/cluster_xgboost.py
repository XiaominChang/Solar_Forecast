from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import csv
from tensorflow import keras
import matplotlib.pyplot as plt
from hyperopt import fmin, tpe, hp, partial, Trials, STATUS_OK
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, zero_one_loss
import xgboost as xgb
from sklearn.model_selection import train_test_split
import pandas as pd
import graphviz
import shap
import time
from math import sqrt
from sklearn.metrics import mean_squared_error, zero_one_loss,mean_absolute_error,r2_score
from matplotlib.backends.backend_pdf import PdfPages
'''
def xgBoost():
 n_steps=16
    X, y= sequence(n_steps)
    print(X.shape)
    print(y.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    model=xgb.XGBRegressor(max_depth=3, learning_rate=0.3,verbosity=2, n_estimators=200,objective='reg:squarederror', booster='gbtree', n_jobs=4, subsample=0.7, colsample_bytree=0.7 )
    train=(X_train, y_train)
    test=(X_test, y_test)
    model.fit(X_train, y_train, eval_metric=['rmse'],eval_set=[train, test], verbose=True, early_stopping_rounds=10,callbacks=None)
    model.save_model('/home/xcha8737/Solar_Forecast/trainning_data/SolarPrediction.csv/xgboost.model')
    results=model.evals_result()
    epochs=len(results['validation_0']['rmse'])
    x_axis=range(0,epochs)
    fig, ax=plt.subplots()

    ax.plot(x_axis, results['validation_0']['rmse'], label='Train')
    ax.plot(x_axis, results['validation_1']['rmse'], label='Test')
    ax.legend()
    plt.grid(True)
    plt.ylabel('Error')
    plt.title('XGBoost MSE loss')
    plt.show()'''



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
    # print(X.shape)
    # print(Y.shape)
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
        output.append(seq_y)
    #print(np.shape(input))
    #print(np.shape(output))
    return np.array(input), np.array(output)
# n_steps=9
# X,Y=dataReader()
# X, y= sequence(n_steps)

#X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
#y = (Y - Y.min(axis=0)) / (Y.max(axis=0) - Y.min(axis=0))
#print(X.shape)
#print(y.shape)

# data=pd.read_csv("/home/xcha8737/Desktop/test_data/all_data.csv")
# #X=data[["airtemp", "humidity", "insolation", "windspeed", "winddirection"]]
# X=data[["airtemp", "humidity", "insolation"]]
# Y=data['power (W)']

pdata=pd.read_csv('/home/xcha8737/Downloads/cap/dataclean/labels_11.csv')
data=np.array(pdata)
X=data[:,:-1]
y=data[:,-1]

x_train_all, x_predict, y_train_all, y_predict = train_test_split(X, y, test_size=0.10, random_state=100)
x_train, x_test, y_train, y_test = train_test_split(x_train_all, y_train_all, test_size=0.2, random_state=100)


# x_train_all, x_predict, y_train_all, y_predict = train_test_split(X, Y, test_size=0.10, random_state=100)
# x_train, x_test, y_train, y_test = train_test_split(x_train_all, y_train_all, test_size=0.2, random_state=100)

# x_train_all, x_predict, y_train_all, y_predict = train_test_split(X, y, test_size=0.10, random_state=100)
#
# x_train, x_test, y_train, y_test = train_test_split(x_train_all, y_train_all, test_size=0.2, random_state=100)

#x_train_all=pd.DataFrame(x_train_all,columns=["ghi", "ghi90","ghi10", "ebh", "dni", "dni10", "dni90", "dhi", "air_temp", "zenith", "azimuth", "cloud_opacity"])
#x_predict=pd.DataFrame(x_predict,columns=["ghi", "ghi90","ghi10", "ebh", "dni", "dni10", "dni90", "dhi", "air_temp", "zenith", "azimuth", "cloud_opacity"])
#x_train=pd.DataFrame(x_train,columns=["ghi", "ghi90","ghi10", "ebh", "dni", "dni10", "dni90", "dhi", "air_temp", "zenith", "azimuth", "cloud_opacity"])
#x_test=pd.DataFrame(x_test,columns=["ghi", "ghi90","ghi10", "ebh", "dni", "dni10", "dni90", "dhi", "air_temp", "zenith","azimuth", "cloud_opacity"])


#dtrain = xgb.DMatrix(data=x_train,label=y_train,missing=-999.0,feature_names=["AirTemp", "Azimuth","CloudOpacity", "DewpointTemp", "Dhi", "Dni", "Ebh", "Ghi", "PrecipitableWater", "RelativeHumidity", "SurfacePressure", "WindDirection10m","WindSpeed10m", "Zenith"])
#dtest = xgb.DMatrix(data=x_test,label=y_test,missing=-999.0, feature_names=["AirTemp", "Azimuth","CloudOpacity", "DewpointTemp", "Dhi", "Dni", "Ebh", "Ghi", "PrecipitableWater", "RelativeHumidity", "SurfacePressure", "WindDirection10m","WindSpeed10m", "Zenith"])
# dtrain = xgb.DMatrix(data=x_train,label=y_train,missing=-999.0)
# dtest = xgb.DMatrix(data=x_test,label=y_test,missing=-999.0)
# evallist = [(dtrain, 'train'),(dtest, 'eval')]
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
         'time_step':hp.randint('time_step',13)
         }
def argsDict_tranform(argsDict, isPrint=False):
    argsDict["max_depth"] = argsDict["max_depth"] + 5
    argsDict['n_estimators'] = argsDict['n_estimators'] + 100
    argsDict["learning_rate"] = argsDict["learning_rate"] * 0.02 + 0.05
    argsDict["min_child_weight"] = argsDict["min_child_weight"] + 5
    argsDict['time_step']=argsDict['time_step']+1
    return argsDict

def xgboost_factory(argsDict):
    argsDict=argsDict_tranform(argsDict)
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
    # params = {'nthread': 4,
    #           'max_depth': 13,
    #           'n_estimators': 291,
    #           'eta': 0.051,
    #           'subsample': 0.215,
    #           'min_child_weight': 16,
    #           'objective': 'reg:squarederror',
    #           'verbosity': 0,
    #           'gamma': 0.633,
    #           'colsample_bytree': 0.894,
    #           'alpha': 0,
    #           'lambda': 0.1109,
    #           'scale_pos_weight': 0,
    #           'seed': 100,
    #           'missing': -999,
    #           "booster" : 'gbtree'
    #           }
    params['eval_metric'] = ['rmse']

    xrf = xgb.train(params, dtrain,argsDict['n_estimators'], evallist, early_stopping_rounds=10)
    #xrf = xgb.train(params, dtrain, num_boost_round=argsDict['n_estimators'], evallist, early_stopping_rounds=10)
    loss=get_tranformer_score(xrf,x_predict,y_predict)

    return {'loss': loss, 'status': STATUS_OK}

def get_tranformer_score(tranformer,x_predict, y_predict):
    xrf = tranformer
    dpredict = xgb.DMatrix(x_predict)
    prediction = xrf.predict(dpredict, ntree_limit=xrf.best_ntree_limit)
    # prediction = prediction * ((Y.max(axis=0) - Y.min(axis=0))) + Y.min(axis=0)
    # y_predict1 = y_predict * ((Y.max(axis=0) - Y.min(axis=0))) + Y.min(axis=0)
    return np.sqrt(mean_squared_error(y_predict, prediction))


def xgbest_train(argsDict):
    argsDict=argsDict_tranform(argsDict)
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
    # params = {'nthread': 4,
    #           'max_depth': 13,
    #           'n_estimators': 291,
    #           'eta': 0.051,
    #           'subsample': 0.215,
    #           'min_child_weight': 16,
    #           'objective': 'reg:squarederror',
    #           'verbosity': 0,
    #           'gamma': 0.633,
    #           'colsample_bytree': 0.894,
    #           'alpha': 0,
    #           'lambda': 0.1109,
    #           'scale_pos_weight': 0,
    #           'seed': 100,
    #           'missing': -999,
    #           "booster" : 'gbtree'
    #           }
    params['eval_metric'] = ['rmse']



    xrf = xgb.train(params, dtrain, argsDict['n_estimators'], evallist, early_stopping_rounds=10)
    loss=get_tranformer_score(xrf,x_predict,y_predict)
    xrf.save_model('/home/xcha8737/Solar_Forecast/trainning_data/dataclean/dataclean/xgboost0021.model')
    '''print(xrf.feature_names)
    xgb.plot_importance(xrf)
    plt.show()
    shap_value=shap.TreeExplainer(xrf).shap_values(x_train_all)
    shap.summary_plot(shap_value, x_train_all, plot_type="bar")
    #fig=plt.gcf()
    plt.show()'''
    time_start=time.time()
    dpredict = xgb.DMatrix(x_predict)
    result=xrf.predict(dpredict)
    time_end=time.time()
    print('totally cost', time_end - time_start)
    print("rmse is ：", sqrt(mean_squared_error(y_predict, result)))
    print("mae is ：", mean_absolute_error(y_predict, result))
    print('r2 is :', r2_score(y_predict, result))

    return {'loss': loss, 'status': STATUS_OK}


trials = Trials()
algo = partial(tpe.suggest, n_startup_jobs=20)
best = fmin(xgboost_factory, space, algo=algo, max_evals=200, pass_expr_memo_ctrl=None, trials=trials)
time_start=time.time()
MSE = xgbest_train(best)
time_end=time.time()
print('training cost is: ', time_end-time_start)
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

# xs0 = [t['misc']['vals']['time_step'][0] for t in trials.trials]
# ys=[t['result']['loss'] for t in trials.trials]
# xs0=np.array(xs0)
# ys=np.array(ys)
# xs0=xs0.reshape([-1,1])
# ys=ys.reshape([-1,1])
# data=np.hstack((xs0,ys))
# pandata=pd.DataFrame(data,columns=['x','y'])
# pandata.to_csv('/home/xcha8737/Downloads/cap/dataclean/test_timestep_xgboost.csv')
# plt.figure()
# # plt.plot(xs0, ys, color='blue',   label='time_step')
# plt.scatter(xs0, ys,  s=20, color='blue',  label='time_step')
# #plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
# plt.grid(True)
# plt.xlabel('time_step')
# plt.ylabel('loss')
# plt.legend(loc="upper right")
# plt.savefig('/home/xcha8737/Solar_Forecast/trainning_data/dataclean/dataclean/xgboost_result.pdf',format = 'pdf', dpi = 1000, bbox_inches = 'tight')
# plt.show()

'''plt.figure()
plt.scatter(xs0, ys,  s=20, color='darkorange', linewidth=0.01, alpha=0.75, label='learning rate')
plt.grid(True)
plt.xlabel('learning rate')
plt.ylabel('loss')
plt.legend(loc="upper right")
plt.show()

plt.figure()
plt.scatter(xs1, ys, s=20,  color='r',linewidth=0.01, alpha=0.75, label='n_estimators')
#plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
plt.grid(True)
plt.xlabel('n_estimatiors')
plt.ylabel('loss')
plt.legend(loc="upper right")
plt.show()

plt.figure()
plt.scatter(xs2, ys, s=20, color='blue', linewidth=0.01, alpha=0.75, label='max_depth')
#plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
plt.grid(True)
plt.xlabel('max_depth')
plt.ylabel('loss')
plt.legend(loc="upper right")
plt.show()

plt.figure()
plt.scatter(xs3, ys, s=20, color='g', linewidth=0.01, alpha=0.75, label='min_child_weight')
#plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
plt.grid(True)
plt.xlabel('min_child_weight')
plt.ylabel('loss')
plt.legend(loc="upper right")
plt.show()'''




