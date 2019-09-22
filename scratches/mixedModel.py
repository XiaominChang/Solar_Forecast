from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import math
import csv
from tensorflow import keras
import matplotlib.pyplot as plt
from hyperopt import fmin, tpe, hp, partial, Trials, STATUS_OK,STATUS_FAIL
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, zero_one_loss
import xgboost as xgb
from sklearn.model_selection import train_test_split
import xgboost as xgb
import dill
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


'''def sequence1( n_steps):
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
    return np.array(input), np.array(output)'''


def sequence( n_steps):
    X,Y=dataReader()
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
n_steps=4
X, y= sequence(n_steps)
xgboost=xgb.Booster()
f=open('C:/Users/chang/Documents/GitHub/dataclean/dataclean/GRNN.dill', 'rb')
xgboost.load_model('C:/Users/chang/Documents/GitHub/dataclean/dataclean/xgboost_test.model')
grnn=dill.load(f)
#gru=keras.models.load_model('C:/Users/chang/Documents/GitHub/dataclean/dataclean/GRU.h5')
svr=joblib.load('C:/Users/chang/Documents/GitHub/dataclean/dataclean/model_svr.pkl')

x_train_all, x_predict, y_train_all, y_predict = train_test_split(X, y, test_size=0.10, random_state=100)
x_train, x_test, y_train, y_test = train_test_split(x_train_all, y_train_all, test_size=0.2, random_state=100)
x_train_stack=x_train_all.reshape(-1,52)
x_predict_stack=x_predict.reshape(-1,52)
dpredict_train = xgb.DMatrix(x_train_stack)
dpredict_predict = xgb.DMatrix(x_predict_stack)

result1=xgboost.predict(dpredict_train)
result2=grnn.predict(x_train_stack)
#result3=gru.predict(x_train_all)
result3=svr.predict(x_train_stack)
result1=result1.reshape(-1,1)
result3=result3.reshape(-1,1)

result=np.hstack([result1,result2,result3])

x_predict1=xgboost.predict(dpredict_predict)
x_predict2=grnn.predict(x_predict_stack)
#x_predict3=gru.predict(x_predict)
x_predict3=svr.predict(x_predict_stack)

x_predict1=x_predict1.reshape(-1,1)
x_predict3=x_predict3.reshape(-1,1)

result_predict=np.hstack([x_predict1,x_predict2,x_predict3])
print(result.shape)
print(result_predict.shape)

space = {'layer1_output': hp.randint('layer1_output', 30),
         'batch_size': hp.randint('batch_size', 100),
         "lr": hp.uniform('lr', 1e-9, 1e-3),
         "decay": hp.uniform('decay', 1e-9, 1e-3),
         'epochs': hp.randint('epochs', 150),
         "dropout": hp.uniform('dropout', 0, 1),
         }

def argsDict_tranform(argsDict):
    argsDict["layer1_output"] = argsDict["layer1_output"] + 5
    argsDict['batch_size']=argsDict['batch_size']+32
    argsDict['epochs']=argsDict['epochs']+50
    return argsDict

def weight_training(argsDic):
    argsDic=argsDict_tranform(argsDic)
    model=keras.models.Sequential()
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(units=argsDic['layer1_output'],input_shape=(-1,3), activation='sigmoid'))
    model.add(keras.layers.Dropout(argsDic['dropout']))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(1))
    adam = keras.optimizers.Adam(lr=argsDic['lr'], decay=argsDic['decay'])
    model.compile(optimizer=adam, loss='mean_squared_error', metrics=['mae'])
    print('start training')
    model.fit(result,y_train_all, epochs=argsDic['epochs'], batch_size=argsDic['batch_size'], validation_split=0.2)
    loss=get_tranformer_score(model)
    if(loss==10):
        return {'loss':loss, 'status':STATUS_FAIL}
    #model.save('/home/xcha8737/Solar_Forecast/trainning_data/SolarPrediction.csv/GRU.h5')
    else:
        return {'loss':loss, 'status':STATUS_OK}


def get_tranformer_score(tranformer):
    mlp = tranformer
    prediction = mlp.predict(result_predict)
    for i in prediction:
        if math.isnan(i[0]):
            print('nan number is found')
            return 10
    return mean_squared_error(y_predict, prediction.reshape(-1,1))

def weight_training_best(argsDic):
    argsDic=argsDict_tranform(argsDic)
    model=keras.models.Sequential()
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(units=argsDic['layer1_output'],input_shape=(-1,3), activation='sigmoid'))
    model.add(keras.layers.Dropout(argsDic['dropout']))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(1))
    adam = keras.optimizers.Adam(lr=argsDic['lr'], decay=argsDic['decay'])
    model.compile(optimizer=adam, loss='mean_squared_error', metrics=['mae'])
    print('start training')
    model.fit(result,y_train_all, epochs=argsDic['epochs'], batch_size=argsDic['batch_size'], validation_split=0.2)
    model.save('C:/Users/chang/Documents/GitHub/dataclean/dataclean/mixed_model.h5')
    return {'loss':get_tranformer_score(model), 'status':STATUS_OK}

trials = Trials()
algo = partial(tpe.suggest, n_startup_jobs=10)
best = fmin(weight_training, space, algo=algo, max_evals=100, pass_expr_memo_ctrl=None, trials=trials)
MSE = weight_training_best(best)
print('best :', best)
print('best param after transform :')
print(argsDict_tranform(best))
print('\nrmse of the best gru:', np.sqrt(MSE['loss']))
xs0 = [t['misc']['vals']['lr'][0] for t in trials.trials]
xs1 = [t['misc']['vals']['decay'][0] for t in trials.trials]
xs2 = [t['misc']['vals']['layer1_output'][0] for t in trials.trials]
ys=[t['result']['loss'] for t in trials.trials]

plt.figure()
plt.scatter(xs0, ys,  s=20, color='darkorange', linewidth=0.01, alpha=0.75, label='learning rate')
#plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
plt.grid(True)
plt.xlabel('learning rate')
plt.ylabel('loss')
plt.legend(loc="upper right")
plt.show()

plt.figure()
plt.scatter(xs1, ys, s=20,  color='r',linewidth=0.01, alpha=0.75, label='decay')
#plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
plt.grid(True)
plt.xlabel('decay')
plt.ylabel('loss')
plt.legend(loc="upper right")
plt.show()

plt.figure()
plt.scatter(xs2, ys, s=20, color='blue', linewidth=0.01, alpha=0.75, label='layer1_output')
#plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
plt.grid(True)
plt.xlabel('layer1_output')
plt.ylabel('loss')
plt.legend(loc="upper right")
plt.show()

