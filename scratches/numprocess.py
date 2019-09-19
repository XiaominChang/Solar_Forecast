import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt

list=[2, 1, 9, 3,2,4,5,7,3]
x=np.array(list)
x=x.reshape(3,3)
print(x)
'''y=np.arange(10,19)
y=y.reshape(3,3)
print(y)
print(np.hstack([x,y]))

c=np.array([1,2,3,4,5])
print(c)
print(c-c.mean(axis=0))
print(c.mean(axis=0))'''

print(np.argmax(x,1))