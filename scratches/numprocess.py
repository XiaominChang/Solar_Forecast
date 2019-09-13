import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt


xs0=[4,3,5,6,4,5]
xs2=[1,3,4,2,4,5]
ys=[2,3,4,1,3,4]
plt.figure()
plt.scatter(xs0, ys,  s=20, color='darkorange', linewidth=0.01, alpha=0.75, label='penelty factor')
#plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
plt.grid(True)
plt.xlabel('penelty factor')
plt.ylabel('loss')
plt.legend(loc="upper right")
plt.show()

'''plt.figure()
plt.scatter(ys1, xs1, s=20,  color='r',linewidth=0.01, alpha=0.75, label='kernel_type')
#plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
plt.grid(True)
plt.xlabel('time step')
plt.ylabel('kernel_type')
plt.legend(loc="upper right")
plt.show()'''

plt.figure()
plt.scatter(xs2, ys, s=20, color='blue', linewidth=0.01, alpha=0.75, label='gamma')
#plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
plt.grid(True)
plt.xlabel('gamma')
plt.ylabel('loss')
plt.legend(loc="upper right")
plt.show()

