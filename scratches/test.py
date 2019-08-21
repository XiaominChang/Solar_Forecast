import csv
import numpy
import joblib
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

def dataReader():
    file=open(" ", 'r', encoding='utf-8' )
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


def SVRtrain():
    ###############################################################################
    # Generate sample data
    #X = np.sort(5 * np.random.rand(40, 1), axis=0)  # 产生40组数据，每组一个数据，axis=0决定按列排列，=1表示行排列
    #y = np.sin(X).ravel()  # np.sin()输出的是列，和X对应，ravel表示转换成行

    ###############################################################################
    # Add noise to targets
    #y[::5] += 3 * (0.5 - np.random.rand(8))

    ###############################################################################
    # Fit regression model
    input,output=dataReader()
    X=input
    y=output

    #svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    #svr_lin = SVR(kernel='linear', C=1e3)
    svr_poly = SVR(kernel='poly', C=1e3, degree=2)
    print('start trainning')
    #y_rbf = svr_rbf.fit(X, y).predict(X)
    #y_lin = svr_lin.fit(X, y).predict(X)
    y_poly = svr_poly.fit(X, y).predict(X)
    print('finish trainning')
    #joblib.dump(svr_rbf,'C:/Users/Xiaoming/Desktop/trainning_data/model_rbf.pkl')
    #joblib.dump(svr_lin,'C:/Users/Xiaoming/Desktop/trainning_data/model_lin001.pkl')
    joblib.dump(svr_poly,'C:/Users/Xiaoming/Desktop/trainning_data/model_poly001.pkl')
    ###############################################################################
    # look at the results
    lw = 2
    plt.scatter(X[...,0], y, color='darkorange', label='data')
    plt.hold('on')
    #plt.plot(X[...,0], y_rbf, color='blue', lw=lw, label='RBF model')
    #plt.plot(X, y_lin, color='c', lw=lw, label='Linear model')
    plt.plot(X, y_poly, color='cornflowerblue', lw=lw, label='Polynomial model')
    plt.xlabel('data')
    plt.ylabel('target')
    plt.title('Support Vector Regression')
    plt.legend()
    plt.show()
SVRtrain()
print('finish')



