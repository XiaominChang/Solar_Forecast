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
        feature.append(float(row[5]))
        feature.append(float(row[6]))
        feature.append(float(row[9]))
        feature.append(float(row[13]))
        features.append(feature)
        #features.append(row[4:9])
    file.close()
    X=np.array(features)
    return X
x=dataReader()
print(x[0:20])