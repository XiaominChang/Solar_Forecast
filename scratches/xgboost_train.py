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
import graphviz
'''def dataReader():
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
    #x=(X-X.mean(axis=0))/X.std(axis=0)
    x=X
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
    return np.array(input), np.array(output)'''
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


x_train_all, x_predict, y_train_all, y_predict = train_test_split(X, y, test_size=0.10, random_state=100)

x_train, x_test, y_train, y_test = train_test_split(x_train_all, y_train_all, test_size=0.2, random_state=100)

dtrain = xgb.DMatrix(data=x_train,label=y_train,missing=-999.0)
dtest = xgb.DMatrix(data=x_test,label=y_test,missing=-999.0)

evallist = [(dtrain, 'train'),(dtest, 'eval')]

space = {"max_depth": hp.randint("max_depth", 10),
         "n_estimators": hp.randint("n_estimators", 200),
         'learning_rate': hp.uniform('learning_rate', 1e-3, 5e-1),
         "min_split_loss": hp.uniform("min_split_loss", 0, 1),
         "subsample": hp.uniform("subsample", 0,1),
         "colsample_bytree": hp.uniform("colsample_bytree", 0, 1),
         "min_child_weight": hp.randint("min_child_weight", 10),
         "objective": 'reg:squarederror',
         "booster" : 'gbtree',
         "lambda": hp.uniform('lambda', 0, 1)
         }
def argsDict_tranform(argsDict, isPrint=False):
    argsDict["max_depth"] = argsDict["max_depth"] + 5
    argsDict['n_estimators'] = argsDict['n_estimators'] + 100
    argsDict["learning_rate"] = argsDict["learning_rate"] * 0.02 + 0.05
    argsDict["min_child_weight"] = argsDict["min_child_weight"] + 5
    return argsDict

def xgboost_factory(argsDict):
    argsDict = argsDict_tranform(argsDict)

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
              }
    params['eval_metric'] = ['rmse']

    xrf = xgb.train(params, dtrain, params['n_estimators'], evallist, early_stopping_rounds=100)
    loss=get_tranformer_score(xrf)

    return {'loss': loss, 'status': STATUS_OK}

def get_tranformer_score(tranformer):
    xrf = tranformer
    dpredict = xgb.DMatrix(x_predict)
    prediction = xrf.predict(dpredict, ntree_limit=xrf.best_ntree_limit)
    return mean_squared_error(y_predict, prediction)


def xgbest_train(argsDict):
    argsDict = argsDict_tranform(argsDict)

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
              }
    params['eval_metric'] = ['rmse']

    xrf = xgb.train(params, dtrain, params['n_estimators'], evallist, early_stopping_rounds=100)
    loss=get_tranformer_score(xrf)
    xrf.save_model('C:/Users/chang/Documents/GitHub/dataclean/dataclean/xgboost_test.model')
    xgb.plot_tree(xrf, num_trees=5)
    plt.show()
    return {'loss': loss, 'status': STATUS_OK}


trials = Trials()
algo = partial(tpe.suggest, n_startup_jobs=10)
best = fmin(xgboost_factory, space, algo=algo, max_evals=100, pass_expr_memo_ctrl=None, trials=trials)
MSE = xgbest_train(best)
print('best :', best)
print('best param after transform :')
print(argsDict_tranform(best,isPrint=True))
print('\nrmse of the best xgboost:', np.sqrt(MSE['loss']))
xs0 = [t['misc']['vals']['learning_rate'][0] for t in trials.trials]
xs1 = [t['misc']['vals']['n_estimators'][0] for t in trials.trials]
xs2 = [t['misc']['vals']['max_depth'][0] for t in trials.trials]
xs3=[t['misc']['vals']['min_child_weight'][0] for t in trials.trials]
ys=[t['result']['loss'] for t in trials.trials]


plt.figure()
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
plt.show()




