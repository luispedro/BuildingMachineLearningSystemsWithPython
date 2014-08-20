# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

# This script fits several forms of penalized regression

from __future__ import print_function
import numpy as np
from sklearn.cross_validation import KFold
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.datasets import load_boston
boston = load_boston()
x = boston.data
y = boston.target

for name, met in [
        ('elastic-net(.5)', ElasticNet(fit_intercept=True, alpha=0.5)),
        ('lasso(.5)', Lasso(fit_intercept=True, alpha=0.5)),
        ('ridge(.5)', Ridge(fit_intercept=True, alpha=0.5)),
]:
    # Fit on the whole data:
    met.fit(x, y)

    # Predict on the whole data:
    p = met.predict(x)

    e = p - y
    # np.dot(e, e) == sum(ei**2 for ei in e) but faster
    total_error = np.dot(e, e)
    rmse_train = np.sqrt(total_error / len(p))

    # Now, we use 10 fold cross-validation to estimate generalization error
    kf = KFold(len(x), n_folds=10)
    err = 0
    for train, test in kf:
        met.fit(x[train], y[train])
        p = met.predict(x[test])
        e = p - y[test]
        err += np.dot(e, e)

    rmse_10cv = np.sqrt(err / len(x))
    print('Method: {}'.format(name))
    print('RMSE on training: {}'.format(rmse_train))
    print('RMSE on 10-fold CV: {}'.format(rmse_10cv))
    print()
    print()
