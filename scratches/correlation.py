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
import scipy.stats as st
from matplotlib.backends.backend_pdf import PdfPages


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
        #feature.append(float(row[18]))

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
#n_steps=1
#X, y= sequence(n_steps)
#X,Y=dataReader()
#X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
#y = (Y - Y.min(axis=0)) / (Y.max(axis=0) - Y.min(axis=0))





#data=pd.read_csv("/home/xcha8737/Downloads/Training_data/test.csv")
#X=data[["AirTemp", "Azimuth","CloudOpacity", "DewpointTemp", "Dhi", "Dni", "Ebh", "Ghi", "PrecipitableWater", "RelativeHumidity", "SurfacePressure", "WindDirection10m","WindSpeed10m", "Zenith"]]
#Y=data['PV_output']

data=pd.read_csv("/home/xcha8737/Desktop/test_data/all_data.csv")
X=data[["airtemp", "humidity", "insolation", "windspeed", "winddirection"]]
Y=data['power (W)']

print("airtemp: ",st.pearsonr(X["airtemp"],Y))
print("humidity: ",st.pearsonr(X["humidity"],Y))
print("insolation: ",st.pearsonr(X["insolation"],Y))
print("windspeed: ",st.pearsonr(X["windspeed"],Y))
print("winddirection: ",st.pearsonr(X["winddirection"],Y))





'''print("AirTemp",st.pearsonr(X["AirTemp"],Y))
print("Azimuth",st.pearsonr(X["Azimuth"],Y))
print("CloudOpacity",st.pearsonr(X["CloudOpacity"],Y))
print("DewpointTemp",st.pearsonr(X["DewpointTemp"],Y))
print(st.pearsonr(X["Dhi"],Y))
print(st.pearsonr(X["Dni"],Y))
print(st.pearsonr(X["Ebh"],Y))
print(st.pearsonr(X["Ghi"],Y))
print(st.pearsonr(X["PrecipitableWater"],Y))
print(st.pearsonr(X["RelativeHumidity"],Y))
print(st.pearsonr(X["SurfacePressure"],Y))
print(st.pearsonr(X["WindDirection10m"],Y))
print(st.pearsonr(X["WindSpeed10m"],Y))
print(st.pearsonr(X["Zenith"],Y))'''
