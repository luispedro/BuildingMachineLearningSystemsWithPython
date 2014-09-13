# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

from sklearn.linear_model import LinearRegression, Lasso
import numpy as np
from sklearn.datasets import load_boston
import pylab as plt

boston = load_boston()
plt.scatter(boston.data[:, 5], boston.target)
plt.xlabel("Number of rooms (RM)")
plt.ylabel("House Price")


x = boston.data[:, 5]
xmin = x.min()
xmax = x.max()
x = np.transpose(np.atleast_2d(x))
y = boston.target

lr = LinearRegression()
lr.fit(x, y)
plt.plot([xmin, xmax], lr.predict([[xmin], [xmax]]), ':', lw=4, label='OLS model')

las = Lasso()
las.fit(x, y)
plt.plot([xmin, xmax], las.predict([ [xmin], [xmax] ]), '-', lw=4, label='Lasso model')
plt.savefig('Figure3.png', dpi=150)
