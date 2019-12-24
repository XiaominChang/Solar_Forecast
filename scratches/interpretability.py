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
import graphviz

from matplotlib.backends.backend_pdf import PdfPages

pdf = PdfPages('C:/Users/chang/Documents/GitHub/Solar_Forecast/result/XGboost/tree.pdf')
xgboost=xgb.Booster()
xgboost.load_model('C:/Users/chang/Documents/GitHub/dataclean/dataclean/xgboost_test.model')
xgb.plot_tree(xgboost, num_trees=2)
fig = plt.gcf()
fig.set_size_inches(150, 100)
#fig.savefig('C:/Users/chang/Documents/GitHub/Solar_Forecast/result/XGboost/tree.png')
pdf.savefig(fig)
pdf.close()





