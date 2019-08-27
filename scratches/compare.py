from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import numpy as np
import csv
from tensorflow import keras

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
        input.append(seq_x)
        output.append(seq_y)
    print(np.shape(input))
    print(np.shape(output))
    return np.array(input), np.array(output)

#linear=joblib.load('C:/Users/Xiaoming/Desktop/trainning_data/model_lin001.pkl')
#rbf=joblib.load('C:/Users/Xiaoming/Desktop/trainning_data/model_rbf.pkl')
print("start")
#prediction0= linear.predict(X[:1000])
#prediction1= rbf.predict(X[:1000])

#mse_linear0=mean_squared_error(Y[:1000], prediction0)
#mse_linear1=mean_squared_error(Y[:1000], prediction1)
n_steps = 16
X, y = sequence(n_steps)
train_X, train_y = X[:-1000, :], y[:-1000]
test_X, test_y = X[-1000:, :], y[-1000:]

lstm=keras.models.load_model('/home/xcha8737/Solar_Forecast/trainning_data/SolarPrediction.csv/LSTM.h5')
gru=keras.models.load_model('/home/xcha8737/Solar_Forecast/trainning_data/SolarPrediction.csv/GRU.h5')

mse_lstm=mean_squared_error(test_y, lstm.predict(test_X))
mse_gru=mean_squared_error(test_y, gru.predict(test_X))
gru.compile(loss='binary_crossentropy',optimizer='adam')
test=gru.evaluate(test_X,test_y)
print(mse_lstm)
print(mse_gru)
print(test)
