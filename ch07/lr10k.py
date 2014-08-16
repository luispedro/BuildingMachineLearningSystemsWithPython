# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.cross_validation import KFold

USE_EN = False

data, target = load_svmlight_file('data/E2006.train')
if USE_EN:
    lr = ElasticNet(fit_intercept=True)
else:
    lr = LinearRegression(fit_intercept=True)

kf = KFold(len(target), n_folds=10)
err = 0
for train, test in kf:
    lr.fit(data[train], target[train])
    p = lr.predict(data[test])
    p = np.array(p).ravel()
    e = p - target[test]
    err += np.dot(e, e)

rmse_10cv = np.sqrt(err / len(target))


lr.fit(data, target)
p = lr.predict(data)
p = p.ravel()
e = p - target
total_error = np.dot(e, e)
rmse_train = np.sqrt(total_error / len(p))

print('RMSE on training: {}'.format(rmse_train))
print('RMSE on 10-fold CV: {}'.format(rmse_10cv))
