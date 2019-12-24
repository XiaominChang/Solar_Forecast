import dask
import dask.array as da
import dask.dataframe as dd
import distributed.comm.utils
import numpy as np
import xgboost as xgb
from dask.array.utils import assert_eq
from dask.distributed import Client
from sklearn.datasets import load_digits, load_iris
from sklearn.model_selection import train_test_split
import dask_xgboost as dxgb
from operator import add


df=dd.read_csv("/vagrant/Forecast/train_data.csv")
dt=dd.read_csv("/vagrant/Forecast/test_data.csv")

print(df)
train=df[["ghi", "ghi90","ghi10", "ebh", "dni", "dni10", "dni90", "dhi", "air_temp", "zenith", "azimuth", "cloud_opacity"]]
labels=df['pv_estimate']

dtrain = xgb.DMatrix(data=x_train,label=y_train,missing=-999.0,feature_names=["ghi", "ghi90","ghi10", "ebh", "dni", "dni10", "dni90", "dhi", "air_temp", "zenith", "azimuth", "cloud_opacity"])
dtest = xgb.DMatrix(data=x_test,label=y_test,missing=-999.0, feature_names=["ghi", "ghi90","ghi10", "ebh", "dni", "dni10", "dni90", "dhi", "air_temp", "zenith", "azimuth", "cloud_opacity"])

evallist = [(dtrain, 'train'),(dtest, 'eval')]

params = {'nthread': 4,
          'max_depth': 35,
          'n_estimators': 658,
          'eta': 0.051020408174604924,
          'subsample': 0.3088227559116749,
          'min_child_weight': 31,
          'objective': 'reg:squarederror',
          'verbosity': 0,
          'gamma': 0.08148233598388686,
          'colsample_bytree': 0.9515249194846609,
          'alpha': 0,
          'lambda': 0.29415483284369115,
          'scale_pos_weight': 0,
          'seed': 100,
          'missing': -999,
          "booster": 'gbtree'
          }

client=Client('127.0.0.1:8786')
#x=client.submit(add,1,2)
dtrain=dxgb.train(client,params,test,labels,)

