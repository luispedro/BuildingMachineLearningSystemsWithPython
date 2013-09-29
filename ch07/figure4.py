# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

from sklearn.linear_model import Lasso
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
xmin = x.min()
xmax = x.max()
x = np.array([[v, 1] for v in x])
y = boston.target

(slope, bias), res, _, _ = np.linalg.lstsq(x, y)
plt.plot([xmin, xmax], [slope * xmin + bias, slope * xmax + bias], ':', lw=4)

las = Lasso()
las.fit(x, y)
y0 = las.predict([xmin, 1])
y1 = las.predict([xmax, 1])
plt.plot([xmin, xmax], [y0, y1], '-', lw=4)
plt.savefig('Figure3.png', dpi=150)
