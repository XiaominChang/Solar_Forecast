import numpy as np
import pandas as pd
import csv

data1=pd.read_csv("C:/Users/chang/Desktop/trained.csv")
data2=pd.read_csv("C:/Users/chang/Desktop/output.csv")
dataList=[]
'''for i in range(365):
    start=300+i*1440
    end=1140+i*1440
    index=start
    while((index>=start) and (index <=end)):
        dataList.append(data.iloc[index])
        index=index+1'''

#trainData=data.drop_duplicates(['time'],keep='last')
df=pd.merge(data1, data2, how='right', on='time')
df.to_csv("C:/Users/chang/Desktop/all_data.csv")
