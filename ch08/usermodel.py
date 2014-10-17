# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

import numpy as np
from sklearn.linear_model import ElasticNetCV
from load_ml100k import get_train_test
from norm import NormalizePositive
from sklearn import metrics


def main(transpose_inputs=False):
    train,test = get_train_test()
    if transpose_inputs:
        train = train.T
        test = test.T
    binary = (train > 0)
    reg = ElasticNetCV(fit_intercept=True, alphas=[
                       0.0125, 0.025, 0.05, .125, .25, .5, 1., 2., 4.])
    norm = NormalizePositive()
    train = norm.fit_transform(train)

    nusers,nmovies = train.shape

    filled = train.copy()
    for u in range(nusers):
        curtrain = np.delete(train, u, axis=0)
        bu = binary[u]
        reg.fit(curtrain[:,bu].T, train[u, bu])
        filled[u, ~bu] = reg.predict(curtrain[:,~bu].T)
    ifilled = norm.inverse_transform(filled)
    r2 = metrics.r2_score(test[test > 0], ifilled[test > 0])
    print('R2 score (user regression): {:.1%}'.format(r2))

if __name__ == '__main__':
    main()
