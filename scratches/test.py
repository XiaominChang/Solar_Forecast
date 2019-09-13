import csv
import numpy
import joblib
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

'''def dataReader0000():
    file=open("C:/Users/Xiaoming/Desktop/trainning_data/SolarPrediction.csv/SolarPrediction.csv", 'r', encoding='utf-8' )
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
    return x, y'''
def dataReader():
    file=open("/home/xcha8737/Solar_Forecast/trainning_data/SolarPrediction.csv/SolarPrediction.csv", 'r', encoding='utf-8' )
    reader=csv.reader(file)
    features=[]
    output=[]
    for row in reader:
        feature=[]
        #feature.append(row[1])
        #feature.append(row[2])
        feature.append(float(row[3]))
        feature.append(float(row[4]))
        feature.append(float(row[5]))
        feature.append(float(row[6]))
        feature.append(float(row[7]))
        feature.append(float(row[8]))
        features.append(feature)
        #features.append(row[4:9])
        output.append(float(row[3]))
    file.close()
    sep_in=[]
    sep_out=[]
    oct_in=[]
    oct_out=[]
    Nov_in=[]
    Nov_out=[]
    Dec_in=[]
    Dec_out=[]
    i=7416
    print(features[0])
    while(i>-1):
        sep_in.append(features[i])
        sep_out.append(output[i])
        i-=1
    i=16237
    while(i>7416):
        oct_in.append(features[i])
        oct_out.append(output[i])
        i-=1
    i=24521
    while(i>16237):
        Nov_in.append(features[i])
        Nov_out.append(output[i])
        i-=1
    i=len(features)-1
    while(i>24521):
        Dec_in.append(features[i])
        Dec_out.append(output[i])
        i-=1
    input=sep_in + oct_in + Nov_in + Dec_in
    output=sep_out+ oct_out +Nov_out+ Dec_out
    print(len(input))
    print(len(output))
    X=np.array(input)
    Y=np.array(output)
    return X, Y

#x,y=dataReader()
def sequence( n_steps):
    x,y=dataReader()
    input, output=list(), list()
    for i in range(len(x)):
        end=i+n_steps
        if end>len(x)-1:
            break
        seq_x, seq_y= x[i:end, :], y[end]
        data_in=np.hstack(seq_x)
        input.append(data_in)
        output.append(seq_y)
    print(np.shape(input))
    print(np.shape(output))
    return np.array(input), np.array(output)
#sequence(16)
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
    X,y=sequence( 16)
    train_X, train_y=X[:-1000,:],y[:-1000]
    test_X, test_y=X[-1000:,:], y[-1000:]
    #svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    svr_lin = SVR(kernel='linear', C=1e3, n_jobs=4)
    #svr_poly = SVR(kernel='poly', C=1e3, degree=2)
    print('start trainning')
    #y_rbf = svr_rbf.fit(X, y).predict(X)
    y_lin = svr_lin.fit(train_X, train_y)
    #y_poly = svr_poly.fit(X, y).predict(X)
    print('finish trainning')
    #joblib.dump(svr_rbf,'C:/Users/Xiaoming/Desktop/trainning_data/model_rbf.pkl')
    joblib.dump(svr_lin,'/home/xcha8737/Solar_Forecast/trainning_data/SolarPrediction.csv/model_lin.pkl')
    #joblib.dump(svr_poly,'C:/Users/Xiaoming/Desktop/trainning_data/model_poly.pkl')
    ###############################################################################
    # look at the results
    '''lw = 2
    plt.scatter(X[...,0], y, color='darkorange', label='data')
    plt.hold('on')
    plt.plot(X[...,0], y_rbf, color='blue', lw=lw, label='RBF model')
    #plt.plot(X, y_lin, color='c', lw=lw, label='Linear model')
    #plt.plot(X, y_poly, color='cornflowerblue', lw=lw, label='Polynomial model')
    plt.xlabel('data')
    plt.ylabel('target')
    plt.title('Support Vector Regression')
    plt.legend()
    plt.show()'''
#SVRtrain()
#print('finish')

def datapro(*,a ,b):
    c=a+b
    return c
arg={'a':5, 'b':6}
print(datapro(**arg))


