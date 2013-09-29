# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

from __future__ import print_function
from sklearn.cross_validation import KFold
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
import numpy as np
from sklearn.datasets import load_boston
boston = load_boston()
x = np.array([np.concatenate((v, [1])) for v in boston.data])
y = boston.target

for name, met in [
        ('elastic-net(.5)', ElasticNet(fit_intercept=True, alpha=0.5)),
        ('lasso(.5)', Lasso(fit_intercept=True, alpha=0.5)),
        ('ridge(.5)', Ridge(fit_intercept=True, alpha=0.5)),
]:
    met.fit(x, y)
    p = np.array([met.predict(xi) for xi in x])
    e = p - y
    total_error = np.dot(e, e)
    rmse_train = np.sqrt(total_error / len(p))

    kf = KFold(len(x), n_folds=10)
    err = 0
    for train, test in kf:
        met.fit(x[train], y[train])
        p = np.array([met.predict(xi) for xi in x[test]])
        e = p - y[test]
        err += np.dot(e, e)

    rmse_10cv = np.sqrt(err / len(x))
    print('Method: {}'.format(name))
    print('RMSE on training: {}'.format(rmse_train))
    print('RMSE on 10-fold CV: {}'.format(rmse_10cv))
    print()
    print()
