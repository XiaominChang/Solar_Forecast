from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import math
import csv
from tensorflow import keras
import matplotlib.pyplot as plt
from hyperopt import fmin, tpe, hp, partial, Trials, STATUS_OK
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, zero_one_loss
import xgboost as xgb
from sklearn.model_selection import train_test_split
import xgboost as xgb
import dill
import graphviz
import shap
import pandas as pd

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


x_train_all, x_predict, y_train_all, y_predict = train_test_split(X, y, test_size=0.10, random_state=100)

x_train, x_test, y_train, y_test = train_test_split(x_train_all, y_train_all, test_size=0.2, random_state=100)

x_train_all=pd.DataFrame(x_train_all,columns=["ghi", "ghi90","ghi10", "ebh", "dni", "dni10", "dni90", "dhi", "air_temp", "zenith", "azimuth", "cloud_opacity"])
x_predict=pd.DataFrame(x_predict,columns=["ghi", "ghi90","ghi10", "ebh", "dni", "dni10", "dni90", "dhi", "air_temp", "zenith", "azimuth", "cloud_opacity"])
x_train=pd.DataFrame(x_train,columns=["ghi", "ghi90","ghi10", "ebh", "dni", "dni10", "dni90", "dhi", "air_temp", "zenith", "azimuth", "cloud_opacity"])
x_test=pd.DataFrame(x_test,columns=["ghi", "ghi90","ghi10", "ebh", "dni", "dni10", "dni90", "dhi", "air_temp", "zenith","azimuth", "cloud_opacity"])


dtrain = xgb.DMatrix(data=x_train,label=y_train,missing=-999.0,feature_names=["ghi", "ghi90","ghi10", "ebh", "dni", "dni10", "dni90", "dhi", "air_temp", "zenith", "azimuth", "cloud_opacity"])
dtest = xgb.DMatrix(data=x_test,label=y_test,missing=-999.0, feature_names=["ghi", "ghi90","ghi10", "ebh", "dni", "dni10", "dni90", "dhi", "air_temp", "zenith", "azimuth", "cloud_opacity"])
xgboost=xgb.Booster()
xgboost.load_model('/home/xcha8737/Solar_Forecast/trainning_data/dataclean/dataclean/xgboost.model')
f=open('/home/xcha8737/Solar_Forecast/trainning_data/dataclean/dataclean/GRNN.dill', 'rb')
grnn=dill.load(f)
svr=joblib.load('/home/xcha8737/Solar_Forecast/trainning_data/dataclean/dataclean/model_svr.pkl')
#xgboost.feature_names=["ghi", "ghi90","ghi10", "ebh", "dni", "dni10", "dni90", "dhi", "air_temp", "zenith", "azimuth", "cloud_opacity"]
#fig,ax = plt.subplots(figsize=(15,15))
#plt.figure()
#print(xgboost.feature_names)
#print(xgboost.feature_types)
#xgb.plot_importance(xgboost)
#xgb.plot_tree(xgboost)
#plt.show()
pdf=PdfPages('/home/xcha8737/Solar_Forecast/trainning_data/dataclean/dataclean/shap_grnn.pdf')
#xgboost.feature_names=["ghi", "ghi90","ghi10", "ebh", "dni", "dni10", "dni90", "dhi", "air_temp", "zenith", "azimuth", "cloud_opacity"]

#xgb.plot_importance(xgboost, importance_type="cover", show_values=False,xlabel='Score').set_yticklabels(['air_temp', 'dni', 'cloud_opacity', 'ebh', 'zenith', 'ghi10', 'dni10', 'dni90','ghi90', 'dhi', 'azimuth',  'ghi'])
#xgb.plot_importance(xgboost, importance_type="gain", show_values=False,xlabel='Score').set_yticklabels(['air_temp', 'dni', 'cloud_opacity', 'dni10', 'zenith', 'azimuth', 'dhi','dni90', 'ghi90', 'ebh', 'ghi10', 'ghi'])

shap_value = shap.TreeExplainer(xgboost).shap_values(x_train_all)
#fig=xgb.plot_importance(xgboost, importance_type="cover", show_values=False, xlabel='Score')
#fig.set_yticklabels=(['ghi', 'ghi10','ebh','ghi90','dni90','dhi','azimuth', 'zenith', 'dni10','cloud_opacity', 'dni', 'air_temp'])

#shap_value=shap.KernelExplainer(grnn.predict, x_predict).shap_values(x_test)
shap.summary_plot(shap_value, x_train_all,  show=False)
plt.grid()
plt.yticks(fontweight="semibold")
plt.xticks(fontweight="semibold")
#axis_cur=plt.sca(ax)
#shap.summary_plot(shap_value, x_train_all, show=False,auto_size_plot=True, plot_type='bar')
#axis_cur=plt.sca()
fig=plt.gcf()
#a=["ghi", "ghi90","ghi10", "ebh", "dni", "dni10", "dni90", "dhi", "air_temp", "zenith", "azimuth", "cloud_opacity"]
#axis_cur=fig.axes
#front={'weight': 'normal', 'size':14}

#fig.set_xticklabels(fontweight="bold")
#fig.set_size_inches(14, 8)
plt.savefig('/home/xcha8737/Solar_Forecast/trainning_data/dataclean/dataclean/xgboost_SHA.pdf',format = 'pdf', dpi = 1000, bbox_inches = 'tight')
#plt.tick_params()
plt.show()




#shap_value=shap.KernelExplainer(svr.predict, x_train_all).shap_values(x_train_all)
#shap.summary_plot(shap_value, x_train_all, plot_type="bar")
#fig=plt.gcf()
#pdf.savefig(fig)
#plt.show()

