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
X,Y=dataReader()
X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
y = (Y - Y.min(axis=0)) / (Y.max(axis=0) - Y.min(axis=0))
print(X.shape)
print(y.shape)

print(st.pearsonr(X[:, 0],Y))
print(st.pearsonr(X[:, 1],Y))
print(st.pearsonr(X[:, 2],Y))
print(st.pearsonr(X[:, 3],Y))
print(st.pearsonr(X[:, 4],Y))
print(st.pearsonr(X[:, 5],Y))
print(st.pearsonr(X[:, 6],Y))
print(st.pearsonr(X[:, 7],Y))
print(st.pearsonr(X[:, 8],Y))
print(st.pearsonr(X[:, 9],Y))
print(st.pearsonr(X[:, 10],Y))
print(st.pearsonr(X[:, 11],Y))
