import csv
import numpy
import joblib
import numpy as np
from sklearn.svm import SVR
#from sklearn.linear_model import LinearSVR as SVR
import matplotlib.pyplot as plt
from hyperopt import fmin, tpe, hp, partial, Trials,STATUS_OK, STATUS_FAIL
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, zero_one_loss
import xgboost as xgb
from sklearn.model_selection import train_test_split
import pandas as pd
import time
from math import sqrt
from sklearn.metrics import mean_squared_error, zero_one_loss,mean_absolute_error,r2_score
import math
'''def dataReader0000():
    file=open("/home/xcha8737/Solar_Forecast/trainning_data/SolarPrediction.csv/SolarPrediction.csv", 'r', encoding='utf-8' )
    reader=csv.reader(file)
    features=[]
    output=[]
    for row in reader:
        feature=[]
        feature.append(float(row[4]))
        feature.append(float(row[5]))
        feature.append(float(row[6]))
        feature.append(float(row[7]))
        feature.append(float(row[8]))
        features.append(feature)
        #features.append(row[4:9])
        output.append(float(row[3]))
    file.close()
    print (len(features))
    print((features[0]))
    print(len(output))
    x=np.array(features)
    y=np.array(output)
    return x, y'''

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
    return np.array(input), np.array(output)
# n_steps=9
# X, Y= sequence(n_steps)
# X=(X-X.min(axis=0))/(X.max(axis=0)-X.min(axis=0))
# y=(Y-Y.min(axis=0))/(Y.max(axis=0)-Y.min(axis=0))
'''x_train_all, x_predict, y_train_all, y_predict = train_test_split(X, y, test_size=0.10, random_state=100)

x_train, x_test, y_train, y_test = train_test_split(x_train_all, y_train_all, test_size=0.2, random_state=100)'''

# X,Y=dataReader()
# X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
# y = (Y - Y.min(axis=0)) / (Y.max(axis=0) - Y.min(axis=0))
# print(X.shape)
# print(y.shape)

pdata=pd.read_csv('/home/xcha8737/Downloads/cap/dataclean/grnnlabels_4.csv')
data=np.array(pdata)
X=data[:,:-1]
Y=data[:,-1]
X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
y = (Y - Y.min(axis=0)) / (Y.max(axis=0) - Y.min(axis=0))
x_train_all, x_predict, y_train_all, y_predict = train_test_split(X, y, test_size=0.10, random_state=100)
x_train, x_test, y_train, y_test = train_test_split(x_train_all, y_train_all, test_size=0.2, random_state=100)

#
# x_train_all, x_predict, y_train_all, y_predict = train_test_split(X, y, test_size=0.10, random_state=100)
#
# x_train, x_test, y_train, y_test = train_test_split(x_train_all, y_train_all, test_size=0.2, random_state=100)

#x_train_all=pd.DataFrame(x_train_all,columns=["ghi", "ghi90","ghi10", "ebh", "dni", "dni10", "dni90", "dhi", "air_temp", "zenith", "azimuth", "cloud_opacity"])
#x_predict=pd.DataFrame(x_predict,columns=["ghi", "ghi90","ghi10", "ebh", "dni", "dni10", "dni90", "dhi", "air_temp", "zenith", "azimuth", "cloud_opacity"])
#x_train=pd.DataFrame(x_train,columns=["ghi", "ghi90","ghi10", "ebh", "dni", "dni10", "dni90", "dhi", "air_temp", "zenith", "azimuth", "cloud_opacity"])
#x_test=pd.DataFrame(x_test,columns=["ghi", "ghi90","ghi10", "ebh", "dni", "dni10", "dni90", "dhi", "air_temp", "zenith","azimuth", "cloud_opacity"])

space = {'C': hp.uniform('C', 0, 10.0),
        'kernel': hp.choice('kernel', ['rbf']),
        'gamma': hp.uniform('gamma', 0, 20.0),
        'degree': hp.randint('degree', 20)
        }
'''def SVRtrain(argsDic):
    print(argsDic['kernel'])
    argsDic['kernel']='rbf'
    if argsDic['kernel']=='rbf':
        print('rbf')
        svr = SVR(kernel=argsDic['kernel'], C=argsDic['C'], gamma=argsDic['gamma'],verbose=True)
        svr.fit(x_train_all, y_train_all)
    elif argsDic['kernel']=='linear':
        print('linear')
        svr = SVR(kernel=argsDic['kernel'], C=argsDic['C'],verbose=True)
        svr.fit(x_train_all, y_train_all)
    else:
        print('poly')
        svr = SVR(kernel=argsDic['kernel'], C=argsDic['C'], gamma=argsDic['gamma'],degree=argsDic['degree'],verbose=True)
        svr.fit(x_train_all, y_train_all)
    return {'loss': get_tranformer_score(svr), 'status': STATUS_OK}

def SVRtrain_best(argsDic):
    if argsDic['kernel']=='rbf':
        svr = SVR(kernel=argsDic['kernel'], C=argsDic['C'], gamma=argsDic['gamma'],verbose=True)
        svr.fit(x_train_all, y_train_all)
    elif argsDic['kernel']=='linear':
        svr = SVR(kernel=argsDic['kernel'], C=argsDic['C'],verbose=True)
        svr.fit(x_train_all, y_train_all)
    else:
        svr = SVR(kernel=argsDic['kernel'], C=argsDic['C'], degree=argsDic['degree'],verbose=True)
        svr.fit(x_train_all, y_train_all)
    joblib.dump(svr, '/home/xcha8737/Solar_Forecast/model_svr.pkl')
    return {'loss': get_tranformer_score(svr), 'status': STATUS_OK}'''
# space = {'std': hp.uniform('std', 0.1, 1.0),
#          'time_step':hp.randint('time_step',13)}


def argsDict_tranform(argsDict):
    argsDict['time_step']=argsDict['time_step']+1
    return argsDict

def SVRtrain(argsDic):
    # argsDic=argsDict_tranform(argsDic)
    # n_steps = argsDic['time_step']
    # X, Y = sequence(n_steps)
    # X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    # y = (Y - Y.min(axis=0)) / (Y.max(axis=0) - Y.min(axis=0))
    # x_train_all, x_predict, y_train_all, y_predict = train_test_split(X, y, test_size=0.10, random_state=100)
    # x_train, x_test, y_train, y_test = train_test_split(x_train_all, y_train_all, test_size=0.2, random_state=100)
    svr = SVR(kernel='rbf', C=argsDic['C'], gamma=argsDic['gamma'],verbose=True)
    # svr = SVR(kernel='rbf', C=1.862, gamma=0.0180, verbose=True)
    svr.fit(x_train_all, y_train_all)
    loss=get_tranformer_score(svr, x_predict, y_predict, Y)
    if(loss==10):
        return {'loss':loss, 'status':STATUS_FAIL}
    #model.save('/home/xcha8737/Solar_Forecast/trainning_data/SolarPrediction.csv/GRU.h5')
    else:
        return {'loss':loss, 'status':STATUS_OK}

def SVRtrain_best(argsDic):
    # argsDic=argsDict_tranform(argsDic)
    # n_steps = argsDic['time_step']
    # X, Y = sequence(n_steps)
    # X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    # y = (Y - Y.min(axis=0)) / (Y.max(axis=0) - Y.min(axis=0))
    # x_train_all, x_predict, y_train_all, y_predict = train_test_split(X, y, test_size=0.10, random_state=100)
    # x_train, x_test, y_train, y_test = train_test_split(x_train_all, y_train_all, test_size=0.2, random_state=100)
    model = SVR(kernel='rbf', C=argsDic['C'], gamma=argsDic['gamma'],verbose=True)
    # model = SVR(kernel='rbf', C=1.862, gamma=0.0180, verbose=True)
    model.fit(x_train_all, y_train_all)
    joblib.dump(model,'/home/xcha8737/Solar_Forecast/trainning_data/dataclean/dataclean/model_svr005.pkl')
    time_start=time.time()
    result=model.predict(x_predict)
    time_end=time.time()
    result = result * ((Y.max(axis=0) - Y.min(axis=0))) + Y.min(axis=0)
    y_real = y_predict * ((Y.max(axis=0) - Y.min(axis=0))) + Y.min(axis=0)
    print('totally cost', time_end - time_start)
    print("rmse is ：", sqrt(mean_squared_error(y_real, result)))
    print("mae is ：", mean_absolute_error(y_real, result))
    print('r2 is :', r2_score(y_real, result))
    loss=get_tranformer_score(model, x_predict, y_predict, Y)
    return {'loss': loss, 'status': STATUS_OK}



def get_tranformer_score(tranformer,x_predict,y_predict, Y):
    svr = tranformer
    result=svr.predict(x_predict)
    for i in result:
        if math.isnan(i):
            print('nan number is found')
            return 10
    result = result * ((Y.max(axis=0) - Y.min(axis=0))) + Y.min(axis=0)
    y_real = y_predict * ((Y.max(axis=0) - Y.min(axis=0))) + Y.min(axis=0)

    return sqrt(mean_squared_error(y_real,result))


trials = Trials()
algo = partial(tpe.suggest, n_startup_jobs=20)
best = fmin(SVRtrain, space, algo=algo, max_evals=500, pass_expr_memo_ctrl=None, trials=trials)
print('best :', best)
time_start=time.time()
MSE = SVRtrain_best(best)
time_end=time.time()
print('training cost is: ', time_end-time_start)
print('best :', best)
print('best param after transform :')
print('rmse of the best svr:', np.sqrt(MSE['loss']))
#print ('trials:')
#for trial in trials.trials:
#    print (trial)
# xs0 = [t['misc']['vals']['time_step'][0] for t in trials.trials]
# ys=[t['result']['loss'] for t in trials.trials]
# xs0=np.array(xs0)
# ys=np.array(ys)
# xs0=xs0.reshape([-1,1])
# ys=ys.reshape([-1,1])
# data=np.hstack((xs0,ys))
# pandata=pd.DataFrame(data,columns=['x','y'])
# #pandata.to_csv('/home/xcha8737/Downloads/cap/dataclean/test_timestep_svr.csv')
#
# plt.figure()
# # plt.plot(xs0, ys, color='blue',   label='time_step')
# plt.scatter(xs0, ys,  s=20, color='blue',  label='time step')
# #plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
# plt.grid(True)
# plt.xlabel('time step')
# plt.ylabel('loss')
# plt.legend(loc="upper right")
# #plt.savefig('/home/xcha8737/Solar_Forecast/trainning_data/dataclean/dataclean/svr_result.pdf',format = 'pdf', dpi = 1000, bbox_inches = 'tight')
# plt.show()

# xs0 = [t['misc']['vals']['C'][0] for t in trials.trials]
# xs2 = [t['misc']['vals']['gamma'][0] for t in trials.trials]
# xs3=[t['misc']['vals']['degree'][0] for t in trials.trials]
# ys=[t['result']['loss'] for t in trials.trials]
# ys1=[t['tid'] for t in trials.trials]
#
# plt.figure()
# plt.scatter(xs0, ys,  s=20, color='darkorange', linewidth=0.01, alpha=0.75, label='penelty factor')
# #plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
# plt.grid(True)
# plt.xlabel('penelty factor')
# plt.ylabel('loss')
# plt.legend(loc="upper right")
# plt.show()
#
# '''plt.figure()
# plt.scatter(ys1, xs1, s=20,  color='r',linewidth=0.01, alpha=0.75, label='kernel_type')
# #plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
# plt.grid(True)
# plt.xlabel('time step')
# plt.ylabel('kernel_type')
# plt.legend(loc="upper right")
# plt.show()'''
#
# plt.figure()
# plt.scatter(xs2, ys, s=20, color='blue', linewidth=0.01, alpha=0.75, label='gamma')
# #plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
# plt.grid(True)
# plt.xlabel('gamma')
# plt.ylabel('loss')
# plt.legend(loc="upper right")
# plt.show()
#
# plt.figure()
# plt.scatter(xs3, ys, s=20, color='g', linewidth=0.01, alpha=0.75, label='degree')
# #plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
# plt.grid(True)
# plt.xlabel('degree')
# plt.ylabel('loss')
# plt.legend(loc="upper right")
# plt.show()





'''def SVRtrain(argsDic):
    #svr_rbf = SVR(kernel=argsDic['kernel'][0], C=argsDic['C'], gamma=argsDic['gamma'])
    #svr_lin = SVR(kernel=argsDic['kernel'][0], C=argsDic['C'])
    svr_poly = SVR(kernel=argsDic['kernel'][0], C=argsDic['C'], degree=C=argsDic['degree'])
    print('start trainning')
    y_rbf = svr_rbf.fit(x_train_all, y_train_all).predict(X)
    y_lin = svr_lin.fit(x_train_all, y_train_all).predict(X)
    svr_poly.fit(x_train_all, y_train_all).predict(X)
    print('finish trainning')
    joblib.dump(svr_rbf,'C:/Users/chang/Documents/GitHub/Solar_Forecast/trainning_data/model_rbf.pkl')
    joblib.dump(svr_lin,'C:/Users/chang/Documents/GitHub/Solar_Forecast/trainning_data/model_lin.pkl')
    joblib.dump(svr_poly,'C:/Users/chang/Documents/GitHub/Solar_Forecast/trainning_data/model_poly.pkl')
    ###############################################################################
    # look at the results
    lw = 2
    #plt.scatter(X[...,0], y, color='darkorange', label='data')
    #plt.hold('on')
    #plt.plot(X[...,0], y_rbf, color='blue', lw=lw, label='RBF model')
    #plt.plot(X, y_lin, color='c', lw=lw, label='Linear model')
    #plt.plot(X, y_poly, color='cornflowerblue', lw=lw, label='Polynomial model')
    plt.xlabel('data')
    plt.ylabel('target')
    plt.title('Support Vector Regression')
    plt.legend()
    plt.show()'''