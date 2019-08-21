import json
import re
file=open("C:/Users/Xiaoming/Desktop/ACM_trans/cap/cap/data_bakcup/pv_actual.json", 'r+', encoding='utf-8' )
fl=open("C:/Users/Xiaoming/Desktop/ACM_trans/cap/cap/data_bakcup/result.json", 'w+', encoding='utf-8' )
for line in file:
    line=re.sub('(ObjectId)','',line)
    line = re.sub('\(', '', line)
    line = re.sub('\)', '', line)
    line=re.sub('(NumberInt)','', line)
    fl.write(line)
    fl.flush()

file.close()