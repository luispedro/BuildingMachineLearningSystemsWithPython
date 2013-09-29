# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

import numpy as np
from sklearn.datasets import load_boston
import pylab as plt
from mpltools import style
style.use('ggplot')

boston = load_boston()
plt.scatter(boston.data[:, 5], boston.target)
plt.xlabel("RM")
plt.ylabel("House Price")


x = boston.data[:, 5]
x = np.array([[v] for v in x])
y = boston.target

slope, res, _, _ = np.linalg.lstsq(x, y)
plt.plot([0, boston.data[:, 5].max() + 1],
         [0, slope * (boston.data[:, 5].max() + 1)], '-', lw=4)
plt.savefig('Figure1.png', dpi=150)

rmse = np.sqrt(res[0] / len(x))
print('Residual: {}'.format(rmse))
