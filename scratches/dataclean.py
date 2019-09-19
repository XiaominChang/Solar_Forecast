
import json
import pandas as pd
import numpy as np
fl=open("/home/xcha8737/Downloads/cap/dataclean/features.json", 'r+', encoding='utf-8' )
features=fl.readlines()
print(type(features))
count=0
feature=''
dataset=[]
for data in features:
    if(count<17):
        feature+=data
        count=count+1
    else:
        a=json.loads(feature)
        dataset.append(a)
        count=1
        feature=''
        feature+=data
fl.close()
featureset=[]
flag=0
for data in dataset:
    for feature in featureset:
        if feature['period_end']==data['period_end']:
            flag=1
            break
    if flag==0:
        featureset.append(data)
    else:
        flag=0
print(len(featureset))

'''features=pd.DataFrame(featureset)
features['period_end']=pd.to_datetime(features['period_end'])
features.sort_values(by='period_end',ascending=True,inplace=True)
print(features)
features.to_csv('C:/Users/chang/Desktop/cap/cap/dataprocess/features001.csv')'''


fl=open("/home/xcha8737/Downloads/cap/dataclean/labels.json", 'r+', encoding='utf-8' )
features1=fl.readlines()
print(type(features1))
count=0
feature=''
dataset1=[]
for data in features1:
    if(count<6):
        feature+=data
        count=count+1
    else:
        a=json.loads(feature)
        dataset1.append(a)
        count=1
        feature=''
        feature+=data
fl.close()
labelset=[]
flag=0
for data in dataset1:
    for feature in labelset:
        if feature['period_end']==data['period_end']:
            flag=1
            break
    if flag==0:
        labelset.append(data)
    else:
        flag=0
print(len(labelset))
'''labels=pd.DataFrame(labelset)

labels['period_end']=pd.to_datetime(labels['period_end'])


labels.sort_values(by='period_end',ascending=True,inplace=True)
print(labels)
labels.to_csv('C:/Users/chang/Desktop/cap/cap/dataprocess/labels001.csv')'''
input=[]
output=[]
for label in labelset:
    for feature in featureset:
        if label['period_end']==feature['period_end']:
            input.append(feature)
            output.append(label)
            break
print(len(input))
print(len(output))

features=pd.DataFrame(input)
labels=pd.DataFrame(output)

df=pd.merge(features, labels, how='right', on='period_end')
df['period_end']=pd.to_datetime(df['period_end'])
df.sort_values(by='period_end', ascending=True, inplace=True)
df.to_csv('/home/xcha8737/Downloads/cap/dataclean/all_data.csv')

df1=pd.concat([features, labels], axis=1)
df1.to_csv('/home/xcha8737/Downloads/cap/dataclean/weather_result.csv')
df2=df[1:]
df2.to_csv('/home/xcha8737/Downloads/cap/dataclean/trainingset.csv')


