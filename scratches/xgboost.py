from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import numpy as np
import csv
from tensorflow import keras
import matplotlib.pyplot as plt
import xgboost

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
    x,y=dataReader()
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

def xgBoost():
    X,y=sequence( 16)
    train_X, train_y=X[:-1000,:],y[:-1000]
    test_X, test_y=X[-1000:,:], y[-1000:]
    model=xgboost.XGBRegressor(max_depth=3, learning_rate=0.1, n_estimators=100,objective='reg:squarederror', booster='gbtree', n_jobs=4 )
    train=[train_X,train_y]
    test=[test_X,test_y]
    model.fit(train_X,train_y,eval_metric=['auc','error'],eval_set=[train, test], verbose=True, early_stopping_rounds=10)
    model.save_model('/home/xcha8737/Solar_Forecast/trainning_data/SolarPrediction.csv/xgboost.model')

xgboost()
