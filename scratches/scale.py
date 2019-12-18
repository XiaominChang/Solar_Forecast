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
import lightgbm as lgb
import dill


f0=open('/home/xcha8737/Solar_Forecast/trainning_data/dataclean/dataclean/grnn004.dill', 'rb')

model0=dill.load(f0)



# def dataReader():
#     file=open("/home/xcha8737/Solar_Forecast/trainning_data/dataclean/dataclean/all_data.csv", 'r', encoding='utf-8' )
#     reader=csv.reader(file)
#     features=[]
#     output=[]
#     for row in reader:
#         if row[2]=='ghi':
#             continue
#         feature=[]
#         feature.append(float(row[2]))
#         feature.append(float(row[3]))
#         feature.append(float(row[4]))
#         feature.append(float(row[5]))
#         feature.append(float(row[6]))
#         feature.append(float(row[7]))
#         feature.append(float(row[8]))
#         feature.append(float(row[9]))
#         feature.append(float(row[10]))
#         feature.append(float(row[11]))
#
#         feature.append(float(row[12]))
#         feature.append(float(row[13]))
#         feature.append(float(row[18]))
#
#         features.append(feature)
#         #features.append(row[4:9])
#         output.append(float(row[18]))
#     file.close()
#     X=np.array(features)
#     Y=np.array(output)
#     return X, Y
#
# #n_steps=1
# #X, y= sequence(n_steps)
# X,Y=dataReader()
# #X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
# #y = (Y - Y.min(axis=0)) / (Y.max(axis=0) - Y.min(axis=0))
# #print(X.shape)
# #print(y.shape)
#
#
#
#
# x_train_all, x_predict, y_train_all, y_predict = train_test_split(X, Y, test_size=0.10, random_state=100)
#
# x_train, x_test, y_train, y_test = train_test_split(x_train_all, y_train_all, test_size=0.2, random_state=100)
#
#
#
# x_train_all=pd.DataFrame(x_train_all,columns=["ghi", "ghi90","ghi10", "ebh", "dni", "dni10", "dni90", "dhi", "air_temp", "zenith", "azimuth", "cloud_opacity","pv_estimate"])
# x_predict=pd.DataFrame(x_predict,columns=["ghi", "ghi90","ghi10", "ebh", "dni", "dni10", "dni90", "dhi", "air_temp", "zenith", "azimuth", "cloud_opacity","pv_estimate"])
# x_train=pd.DataFrame(x_train,columns=["ghi", "ghi90","ghi10", "ebh", "dni", "dni10", "dni90", "dhi", "air_temp", "zenith", "azimuth", "cloud_opacity","pv_estimate"])
# x_test=pd.DataFrame(x_test,columns=["ghi", "ghi90","ghi10", "ebh", "dni", "dni10", "dni90", "dhi", "air_temp", "zenith","azimuth", "cloud_opacity","pv_estimate"])
#
# #x_train.to_csv('/home/xcha8737/env/vagrant_cluster/Forecast/train_data.csv')
# #x_test.to_csv('/home/xcha8737/env/vagrant_cluster/Forecast/test_data.csv')
# dtrain = lgb.Dataset(x_train, y_train)
# dtest = lgb.Dataset(x_test, y_test)