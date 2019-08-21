from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import numpy as np
import csv

def dataReader():
    file=open("C:/Users/Xiaoming/Desktop/trainning_data/SolarPrediction.csv/SolarPrediction.csv", 'r', encoding='utf-8' )
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
    return x, y
X,Y=dataReader()
linear=joblib.load('C:/Users/Xiaoming/Desktop/trainning_data/model_lin001.pkl')
rbf=joblib.load('C:/Users/Xiaoming/Desktop/trainning_data/model_rbf.pkl')
print("start")
prediction0= linear.predict(X[:1000])
prediction1= rbf.predict(X[:1000])

mse_linear0=mean_squared_error(Y[:1000], prediction0)
mse_linear1=mean_squared_error(Y[:1000], prediction1)
print(mse_linear0,mse_linear1)