import csv
import numpy
import joblib
import numpy as np
from sklearn.svm import SVR
#from sklearn.linear_model import LinearSVR as SVR
import matplotlib.pyplot as plt
from hyperopt import fmin, tpe, hp, partial, Trials,STATUS_OK
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, zero_one_loss
import xgboost as xgb
from sklearn.model_selection import train_test_split

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
    print(features[0])
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
    print(len(input))
    print(len(output))
    X=np.array(input)
    Y=np.array(output)
    return X, Y

#x,y=dataReader()
def sequence( n_steps):
    X,Y=dataReader()
    x=(X-X.mean(axis=0))/X.std(axis=0)
    y=(Y-Y.mean(axis=0))/Y.std(axis=0)
    input, output=list(), list()
    for i in range(len(x)):
        end=i+n_steps
        if end>len(x)-1:
            break
        seq_x, seq_y= x[i:end, :], y[end]
        data_in=np.hstack(seq_x)
        input.append(data_in)
        output.append(seq_y)
    print(np.shape(input))
    print(np.shape(output))
    return np.array(input), np.array(output)
n_steps=16
X, y= sequence(n_steps)
print(X.shape)
print(y.shape)
x_train_all, x_predict, y_train_all, y_predict = train_test_split(X, y, test_size=0.10, random_state=100)

x_train, x_test, y_train, y_test = train_test_split(x_train_all, y_train_all, test_size=0.2, random_state=100)
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

def SVRtrain(argsDic):
    svr = SVR(kernel='rbf', C=argsDic['C'], gamma=argsDic['gamma'],verbose=True)
    svr.fit(x_train_all, y_train_all)
    return {'loss': get_tranformer_score(svr), 'status': STATUS_OK}


def SVRtrain_best(argsDic):
    model = SVR(kernel='rbf', C=argsDic['C'], gamma=argsDic['gamma'],verbose=True)
    model.fit(x_train_all, y_train_all)
    joblib.dump(model,'/home/xcha8737/Solar_Forecast/model_svr.pkl')
    return {'loss': get_tranformer_score(model), 'status': STATUS_OK}



def get_tranformer_score(tranformer):
    svr = tranformer
    prediction = svr.predict(x_predict)
    return mean_squared_error(y_predict, prediction)


trials = Trials()
algo = partial(tpe.suggest, n_startup_jobs=10)
best = fmin(SVRtrain, space, algo=algo, max_evals=100, pass_expr_memo_ctrl=None, trials=trials)
print('best :', best)
MSE = SVRtrain_best(best)
print('best :', best)
print('best param after transform :')
print('rmse of the best svr:', np.sqrt(MSE['loss']))
print ('trials:')
#for trial in trials.trials:
#    print (trial)

xs0 = [t['misc']['vals']['C'][0] for t in trials.trials]
'''xs1=[]
for t in trials.trials:
    if t['misc']['vals']['kernel_type'][0] =='rbf':
        xs1.append(0)
    elif t['misc']['vals']['kernel_type'][0] =='linear':
        xs1.append(1)
    else:
        xs1.append(2)'''

'''xs1 = [t['misc']['vals']['kernel_type'][0] for t in trials.trials]'''
xs2 = [t['misc']['vals']['gamma'][0] for t in trials.trials]
xs3=[t['misc']['vals']['degree'][0] for t in trials.trials]
ys=[t['result']['loss'] for t in trials.trials]
ys1=[t['tid'] for t in trials.trials]

plt.figure()
plt.scatter(xs0, ys,  s=20, color='darkorange', linewidth=0.01, alpha=0.75, label='penelty factor')
#plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
plt.grid(True)
plt.xlabel('penelty factor')
plt.ylabel('loss')
plt.legend(loc="upper right")
plt.show()

'''plt.figure()
plt.scatter(ys1, xs1, s=20,  color='r',linewidth=0.01, alpha=0.75, label='kernel_type')
#plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
plt.grid(True)
plt.xlabel('time step')
plt.ylabel('kernel_type')
plt.legend(loc="upper right")
plt.show()'''

plt.figure()
plt.scatter(xs2, ys, s=20, color='blue', linewidth=0.01, alpha=0.75, label='gamma')
#plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
plt.grid(True)
plt.xlabel('gamma')
plt.ylabel('loss')
plt.legend(loc="upper right")
plt.show()

plt.figure()
plt.scatter(xs3, ys, s=20, color='g', linewidth=0.01, alpha=0.75, label='degree')
#plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
plt.grid(True)
plt.xlabel('degree')
plt.ylabel('loss')
plt.legend(loc="upper right")
plt.show()





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