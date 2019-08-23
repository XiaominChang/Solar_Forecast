import numpy as np
from pandas import read_csv


a= np.array([[[[1, 2, 3], [6, 7, 8], [9, 10, 1]],
                [[0, 1, 2], [2, 3, 4], [3, 4, 1]]],

               [[[1, 2, 3], [6, 7, 8], [9, 10, 1]],
                [[0, 1, 2], [2, 3, 4], [3, 4, 1]]]], dtype=np.float32)
print(a.shape)