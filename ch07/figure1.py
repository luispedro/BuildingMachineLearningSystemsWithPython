# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

import numpy as np
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
from mpltools import style
style.use('ggplot')

boston = load_boston()

# Index number five in the number of rooms
plt.scatter(boston.data[:, 5], boston.target)
plt.xlabel("Number of rooms (RM)")
plt.ylabel("House Price")

x = boston.data[:, 5]
# fit (used below) takes a two-dimensional array as input. We use np.atleast_2d
# to convert from one to two dimensional, then transpose to make sure that the
# format matches:
x = np.transpose(np.atleast_2d(x))

y = boston.target

lr = LinearRegression(fit_intercept=False)

lr.fit(x, y)

plt.plot([0, boston.data[:, 5].max() + 1],
         [0, lr.predict(boston.data[:, 5].max() + 1)], '-', lw=4)
plt.savefig('Figure1.png', dpi=150)

# The instance member `residues_` contains the sum of the squared residues
rmse = np.sqrt(lr.residues_ / len(x))
print('RMSE: {}'.format(rmse))
