# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License


# This script plots prediction-vs-actual on training set for the Boston dataset
# using OLS regression

from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
import pylab as plt

boston = load_boston()

x = boston.data
y = boston.target

lr = LinearRegression()
lr.fit(x, y)
p = lr.predict(x)
plt.scatter(p, y)
plt.xlabel('Predicted price')
plt.ylabel('Actual price')
plt.plot([y.min(), y.max()], [y.min(), y.max()])

plt.savefig('Figure4.png', dpi=150)
