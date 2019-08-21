import json
import numpy
fl=open("C:/Users/Xiaoming/Desktop/ACM_trans/cap/cap/data_bakcup/dataset.json", 'r+', encoding='utf-8' )
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
print(len(dataset))
fl.close()
trainset=[]
for data in dataset:
    list=[]
    list.append(data['ghi'])
    list.append(data['ghi90'])
    list.append(data['ghi10'])
    list.append(data['ebh'])
    list.append(data['dni'])
    list.append(data['dni10'])
    list.append(data['dni90'])
    list.append(data['dhi'])
    list.append(data['air_temp'])
    list.append(data['zenith'])
    list.append(data['azimuth'])
    list.append(data['cloud_opacity'])
    list.append(data['period_end'])
    trainset.append(list)
print(len(trainset))


fl=open("C:/Users/Xiaoming/Desktop/ACM_trans/cap/cap/data_bakcup/result.json", 'r+', encoding='utf-8' )
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
print(len(dataset1))
fl.close()
trainset1=[]
for data in dataset1:
    list=[]
    list.append(data['period_end'])
    list.append(data['pv_estimate'])
    trainset1.append(list)
print(len(trainset1))
i=0
while(i<len(trainset1)):



