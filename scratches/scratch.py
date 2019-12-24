import json
import re
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import math
import csv
from tensorflow import keras
import matplotlib.pyplot as plt
from hyperopt import fmin, tpe, hp, partial, Trials, STATUS_OK
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, zero_one_loss,r2_score
import xgboost as xgb
from sklearn.model_selection import train_test_split
import xgboost as xgb
import dill
import datetime
import random
from keras_layer_normalization import LayerNormalization
import lightgbm as lgb
import time
import pandas as pd


# file=open("C:/Users/Xiaoming/Desktop/ACM_trans/cap/cap/data_bakcup/pv_actual.json", 'r+', encoding='utf-8' )
# fl=open("C:/Users/Xiaoming/Desktop/ACM_trans/cap/cap/data_bakcup/result.json", 'w+', encoding='utf-8' )
# for line in file:
#     line=re.sub('(ObjectId)','',line)
#     line = re.sub('\(', '', line)
#     line = re.sub('\)', '', line)
#     line=re.sub('(NumberInt)','', line)
#     fl.write(line)
#     fl.flush()
#
# file
# .close()
data=[51.42, 46.85, 32.58, 35.49, 45.90, 47.85]
labels=['SVR', 'GRNN', 'XGBoost', 'LightBoost', 'LSTM', 'GRU']
#data=[47.55, 55.51, 46.99, 49.30, 50.31, 42.34, 48.39, 46.07, 40.57, 47.21, 52.59, 46.43 ]
#data=[44.01, 51.86, 43.29, 44.27, 48.15, 41.52, 48.04, 45.28, 40.38, 46.66, 50.44, 47.12]
#svr:data=[63.73, 69.87, 58.57, 58.63, 63.03, 63.36, 63.56, 61.48, 59.95, 57.74, 62.03, 56.74]
#grnn: data=[63.53,67.77,57.28,55.80,48.94,52.63,51.01,49.08, 54.29, 56.67, 68.25, 59.36]
data2=[1,2,2,3,4,1]
print(len(data))
#x=[i for i in range(1,13)]
total_width, n=0.8, 2
x=np.arange(1, len(data)+1)
width=total_width/2
x=x - (total_width - width) / 2
fig=plt.figure()
# plt.plot(xs0, ys, color='blue',   label='time_step')
ax1=fig.add_subplot(111)
lins1=ax1.bar( x, data, width=width, tick_label=labels, color='royalblue',  label='RMSE')
#ax1.bar(x + width, data2, width=width, tick_label=labels, color='red', label='time step')
#plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
# plt.grid(True)
ax1.set_xlabel('Methods', fontweight='medium', fontsize='large')
ax1.set_ylabel('RMSE loss', fontweight='medium',fontsize='large')

# for xtick in ax1.get_xticklabels():
#     xtick.set_rotation(10)

# ax1.bar( range(1, len(data)+1), data, tick_label=labels, color='cornflowerblue',  label='time step')
# #plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
# # plt.grid(True)
# ax1.set_xlabel('Methods', fontweight='medium', fontsize='large')
# ax1.set_ylabel('RMSE loss', fontweight='medium',fontsize='large')
#
ax2=ax1.twinx()
lins2=ax2.bar( x+width, data2, width=width, tick_label=labels, color='limegreen',  label='MSE')
#plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
# plt.grid(True)
ax2.set_xlabel('Methods', fontweight='medium', fontsize='large')
ax2.set_ylabel('MSE loss', fontweight='medium',fontsize='large')
# for xtick in ax1.get_xticklabels():
#x_ticks=np.arange(1,13, 1)
#plt.ylim((25, 55))
#plt.xticks(x_ticks, x,  fontweight='semibold')
lns=lins1+lins2
print(lns)
labs=[l.get_label() for l in lns]
# print(labs)

plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
# plt.legend(lns, labs, loc="upper right")
# plt.legend(lns, labs)
plt.legend(loc=1,bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
plt.savefig('/home/xcha8737/Solar_Forecast/trainning_data/dataclean/dataclean/compare_bar.pdf',format = 'pdf', dpi = 1000, bbox_inches = 'tight')
plt.show()
# print(len(data))
# x=[i for i in range(1,13)]
# plt.figure()
# # plt.plot(xs0, ys, color='blue',   label='time_step')
# plt.bar( range(1, len(data)+1), data, color='cornflowerblue',  label='time step')
# #plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
# # plt.grid(True)
# plt.xlabel('time step', fontweight='medium', fontsize='large')
# plt.ylabel('RMSE loss', fontweight='medium',fontsize='large')
#
# x_ticks=np.arange(1,13, 1)
# plt.ylim((35, 60))
# plt.xticks(x_ticks, x,  fontweight='semibold')
# plt.rcParams['xtick.direction'] = 'in'
# plt.rcParams['ytick.direction'] = 'in'
# plt.legend(loc="upper right")
# plt.savefig('/home/xcha8737/Solar_Forecast/trainning_data/dataclean/dataclean/lightgbm_bar.pdf',format = 'pdf', dpi = 1000, bbox_inches = 'tight')
# plt.show()
