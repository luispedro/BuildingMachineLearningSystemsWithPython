import numpy as np
from sklearn.datasets import load_boston
import pylab as plt
from mpltools import style
style.use('ggplot')

boston = load_boston()
plt.scatter(boston.data[:,5], boston.target)
plt.xlabel("RM")
plt.ylabel("House Price")


x = boston.data[:,5]
xmin = x.min()
xmax = x.max()
x = np.array([[v,1] for v in x])
y = boston.target

(slope,bias),res,_,_ = np.linalg.lstsq(x,y)
plt.plot([xmin,xmax],[slope*xmin + bias, slope*xmax + bias], '-', lw=4)
plt.savefig('Figure2.png',dpi=150)

rmse = np.sqrt(res[0]/len(x))
print('Residual: {}'.format(rmse))
