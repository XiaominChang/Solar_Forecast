import numpy as np
from sklearn import datasets, preprocessing
from neupy import algorithms
import csv
import dill
import math
from math import sqrt
import os
from loss import LossHistory
from sklearn.utils import shuffle
from hyperopt import fmin, tpe, hp, partial, Trials, STATUS_OK, STATUS_FAIL
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, zero_one_loss
import matplotlib.pyplot as plt
import joblib
import pandas as pd

def dataReader():
    file=open("/home/xcha8737/Downloads/cap/dataclean/all_data.csv", 'r', encoding='utf-8' )
    reader=csv.reader(file)
    features=[]
    for row in reader:
        if row[2]=='ghi':
            continue
        feature=[]
        feature.append(float(row[2]))
        #feature.append(float(row[5]))
        #feature.append(float(row[6]))
        feature.append(float(row[9]))
        #feature.append(float(row[13]))
        features.append(feature)
        #features.append(row[4:9])
    file.close()
    X=np.array(features)
    return X


def sequence( n_steps):
    X=dataReader()
    print(X.shape)
    x=(X-X.min(axis=0))/(X.max(axis=0)-X.min(axis=0))
    input, output=list(), list()
    #print(x[0:10])
    #print(y[0:10])
    for i in range(len(x)):
        end=i+n_steps
        if end>len(x)-1:
            break
        seq_x= x[i:end, :]
        data_in=np.hstack(seq_x)
        input.append(data_in)
    return np.array(input)

n_steps=4
X= sequence(n_steps)
x_train_all, x_predict= train_test_split(X, test_size=0.10, random_state=100)

def SOMcluster():
    kohonen=algorithms.Kohonen(n_inputs=8, n_outputs=5, step=0.1, verbose=False)
    kohonen.train(x_train_all, epochs=100)
    result=kohonen.predict(X)
    print(result.shape)
    label_sum=np.sum(result, axis=0)
    print(label_sum)
    pd_result=pd.DataFrame(result, columns=['label1', 'label2', 'label3', 'label4','label5'])
    pd_result.to_csv('/home/xcha8737/Downloads/cap/dataclean/labels.csv')
    #nonzero_index=np.argmax(result, 1)
    #pd_label=pd.DataFrame(nonzero_index, columns=['label_index'])
    #pd_label.to_csv('/home/xcha8737/Downloads/cap/dataclean/index.csv')




SOMcluster()