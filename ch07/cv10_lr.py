# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

from sklearn.cross_validation import KFold
from sklearn.linear_model import LinearRegression, ElasticNet
import numpy as np
from sklearn.datasets import load_boston
boston = load_boston()
x = np.array([np.concatenate((v, [1])) for v in boston.data])
y = boston.target
FIT_EN = False

if FIT_EN:
    model = ElasticNet(fit_intercept=True, alpha=0.5)
else:
    model = LinearRegression(fit_intercept=True)
model.fit(x, y)
p = np.array([model.predict(xi) for xi in x])
e = p - y
total_error = np.dot(e, e)
rmse_train = np.sqrt(total_error / len(p))

kf = KFold(len(x), n_folds=10)
err = 0
for train, test in kf:
    model.fit(x[train], y[train])
    p = np.array([model.predict(xi) for xi in x[test]])
    e = p - y[test]
    err += np.dot(e, e)

rmse_10cv = np.sqrt(err / len(x))
print('RMSE on training: {}'.format(rmse_train))
print('RMSE on 10-fold CV: {}'.format(rmse_10cv))
