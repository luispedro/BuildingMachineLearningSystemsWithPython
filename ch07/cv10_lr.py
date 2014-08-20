# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

import numpy as np
from sklearn.cross_validation import KFold
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.datasets import load_boston
boston = load_boston()
x = boston.data
y = boston.target


# Switch this variable to use an Elastic Net instead of OLS
FIT_EN = False

if FIT_EN:
    model = ElasticNet(fit_intercept=True, alpha=0.5)
else:
    model = LinearRegression(fit_intercept=True)

model.fit(x, y)
rmse_train = np.sqrt(model.residues_/len(x))

# Alternatively, we could have computed rmse_train using this expression:
# rmse_train = np.sqrt(np.mean( (model.predict(x) - y) ** 2))
# The results are equivalent

kf = KFold(len(x), n_folds=10)
err = 0
for train, test in kf:
    model.fit(x[train], y[train])
    p = model.predict(x[test])
    e = p - y[test]
    err += np.dot(e, e) # This is the same as np.sum(e * e)

rmse_10cv = np.sqrt(err / len(x))
print('RMSE on training: {}'.format(rmse_train))
print('RMSE on 10-fold CV: {}'.format(rmse_10cv))
