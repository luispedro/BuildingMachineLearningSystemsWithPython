# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

import os
from data import CHART_DIR

import numpy as np
from scipy.stats import norm

from matplotlib import pyplot
np.random.seed(3)

num_per_class = 40
X = np.hstack((norm.rvs(2, size=num_per_class, scale=2),
              norm.rvs(8, size=num_per_class, scale=3)))
y = np.hstack((np.zeros(num_per_class),
               np.ones(num_per_class)))


def lr_model(clf, X):
    return 1.0 / (1.0 + np.exp(-(clf.intercept_ + clf.coef_ * X)))

from sklearn.linear_model import LogisticRegression
logclf = LogisticRegression()
print(logclf)
logclf.fit(X.reshape(num_per_class * 2, 1), y)
print(np.exp(logclf.intercept_), np.exp(logclf.coef_.ravel()))
print("P(x=-1)=%.2f\tP(x=7)=%.2f" %
      (lr_model(logclf, -1), lr_model(logclf, 7)))
X_test = np.arange(-5, 20, 0.1)
pyplot.figure(figsize=(10, 4))
pyplot.xlim((-5, 20))
pyplot.scatter(X, y, c=y)
pyplot.xlabel("feature value")
pyplot.ylabel("class")
pyplot.grid(True, linestyle='-', color='0.75')
pyplot.savefig(
    os.path.join(CHART_DIR, "log_reg_example_data.png"), bbox_inches="tight")


def lin_model(clf, X):
    return clf.intercept_ + clf.coef_ * X

from sklearn.linear_model import LinearRegression
clf = LinearRegression()
print(clf)
clf.fit(X.reshape(num_per_class * 2, 1), y)
X_odds = np.arange(0, 1, 0.001)
pyplot.figure(figsize=(10, 4))
pyplot.subplot(1, 2, 1)
pyplot.scatter(X, y, c=y)
pyplot.plot(X_test, lin_model(clf, X_test))
pyplot.xlabel("feature value")
pyplot.ylabel("class")
pyplot.title("linear fit on original data")
pyplot.grid(True, linestyle='-', color='0.75')

X_ext = np.hstack((X, norm.rvs(20, size=100, scale=5)))
y_ext = np.hstack((y, np.ones(100)))
clf = LinearRegression()
clf.fit(X_ext.reshape(num_per_class * 2 + 100, 1), y_ext)
pyplot.subplot(1, 2, 2)
pyplot.scatter(X_ext, y_ext, c=y_ext)
pyplot.plot(X_ext, lin_model(clf, X_ext))
pyplot.xlabel("feature value")
pyplot.ylabel("class")
pyplot.title("linear fit on additional data")
pyplot.grid(True, linestyle='-', color='0.75')
pyplot.savefig(
    os.path.join(CHART_DIR, "log_reg_log_linear_fit.png"), bbox_inches="tight")

pyplot.figure(figsize=(10, 4))
pyplot.xlim((-5, 20))
pyplot.scatter(X, y, c=y)
pyplot.plot(X_test, lr_model(logclf, X_test).ravel())
pyplot.plot(X_test, np.ones(X_test.shape[0]) * 0.5, "--")
pyplot.xlabel("feature value")
pyplot.ylabel("class")
pyplot.grid(True, linestyle='-', color='0.75')
pyplot.savefig(
    os.path.join(CHART_DIR, "log_reg_example_fitted.png"), bbox_inches="tight")

X = np.arange(0, 1, 0.001)
pyplot.figure(figsize=(10, 4))
pyplot.subplot(1, 2, 1)
pyplot.xlim((0, 1))
pyplot.ylim((0, 10))
pyplot.plot(X, X / (1 - X))
pyplot.xlabel("P")
pyplot.ylabel("odds = P / (1-P)")
pyplot.grid(True, linestyle='-', color='0.75')

pyplot.subplot(1, 2, 2)
pyplot.xlim((0, 1))
pyplot.plot(X, np.log(X / (1 - X)))
pyplot.xlabel("P")
pyplot.ylabel("log(odds) = log(P / (1-P))")
pyplot.grid(True, linestyle='-', color='0.75')
pyplot.savefig(
    os.path.join(CHART_DIR, "log_reg_log_odds.png"), bbox_inches="tight")
