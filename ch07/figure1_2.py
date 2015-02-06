# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

import numpy as np
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib import pyplot as plt

boston = load_boston()

# Index number five in the number of rooms
fig,ax = plt.subplots()
ax.scatter(boston.data[:, 5], boston.target)
ax.set_xlabel("Average number of rooms (RM)")
ax.set_ylabel("House Price")

x = boston.data[:, 5]
# fit (used below) takes a two-dimensional array as input. We use np.atleast_2d
# to convert from one to two dimensional, then transpose to make sure that the
# format matches:
x = np.transpose(np.atleast_2d(x))

y = boston.target

lr = LinearRegression(fit_intercept=False)
lr.fit(x, y)

ax.plot([0, boston.data[:, 5].max() + 1],
         [0, lr.predict(boston.data[:, 5].max() + 1)], '-', lw=4)
fig.savefig('Figure1.png')

mse = mean_squared_error(y, lr.predict(x))
rmse = np.sqrt(mse)
print('RMSE (no intercept): {}'.format(rmse))

# Repeat, but fitting an intercept this time:
lr = LinearRegression(fit_intercept=True)

lr.fit(x, y)

fig,ax = plt.subplots()
ax.set_xlabel("Average number of rooms (RM)")
ax.set_ylabel("House Price")
ax.scatter(boston.data[:, 5], boston.target)
xmin = x.min()
xmax = x.max()
ax.plot([xmin, xmax], lr.predict([[xmin], [xmax]]) , '-', lw=4)
fig.savefig('Figure2.png')

mse = mean_squared_error(y, lr.predict(x))
print("Mean squared error (of training data): {:.3}".format(mse))

rmse = np.sqrt(mse)
print("Root mean squared error (of training data): {:.3}".format(rmse))

cod = r2_score(y, lr.predict(x))
print('COD (on training data): {:.2}'.format(cod))

