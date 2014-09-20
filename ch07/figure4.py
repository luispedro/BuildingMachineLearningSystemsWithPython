# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License


# This script plots prediction-vs-actual on training set for the Boston dataset
# using OLS regression
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
import pylab as plt

boston = load_boston()

x = boston.data
y = boston.target

lr = LinearRegression()
lr.fit(x, y)
p = lr.predict(x)
print("RMSE: {:.2}.".format(np.sqrt(mean_squared_error(y, p))))
print("R2: {:.2}.".format(lr.score(x, y)))
plt.scatter(p, y)
plt.xlabel('Predicted price')
plt.ylabel('Actual price')
plt.plot([y.min(), y.max()], [y.min(), y.max()], lw=4)

plt.savefig('Figure4.png', dpi=150)
