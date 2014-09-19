# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.cross_validation import KFold
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_squared_error, r2_score

data, target = load_svmlight_file('data/E2006.train')

# Edit the lines below if you want to switch method:
# met = LinearRegression(fit_intercept=True)
met = ElasticNetCV()

kf = KFold(len(target), n_folds=5)
pred = np.zeros_like(target)
for train, test in kf:
    met.fit(data[train], target[train])
    pred[test] = met.predict(data[test])

print('[EN 0.1] RMSE on testing (5 fold), {:.2}'.format(np.sqrt(mean_squared_error(target, pred))))
print('[EN 0.1] R2 on testing (5 fold), {:.2}'.format(r2_score(target, pred)))
print('')

met.fit(data, target)
pred = met.predict(data)
print('[EN 0.1] RMSE on training, {:.2}'.format(np.sqrt(mean_squared_error(target, pred))))
print('[EN 0.1] R2 on training, {:.2}'.format(r2_score(target, pred)))


