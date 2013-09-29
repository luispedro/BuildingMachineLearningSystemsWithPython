# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

import numpy as np
from scipy import sparse
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV
from sklearn.cross_validation import KFold

data = np.array([[int(tok) for tok in line.split('\t')[:3]]
                for line in open('data/ml-100k/u.data')])
ij = data[:, :2]
ij -= 1  # original data is in 1-based system
values = data[:, 2]
reviews = sparse.csc_matrix((values, ij.T)).astype(float)

reg = ElasticNetCV(fit_intercept=True, alphas=[
                   0.0125, 0.025, 0.05, .125, .25, .5, 1., 2., 4.])


def movie_norm(xc):
    xc = xc.copy().toarray()
    x1 = np.array([xi[xi > 0].mean() for xi in xc])
    x1 = np.nan_to_num(x1)

    for i in range(xc.shape[0]):
        xc[i] -= (xc[i] > 0) * x1[i]
    return xc, x1


def learn_for(i):
    u = reviews[i]
    us = np.delete(np.arange(reviews.shape[0]), i)
    ps, = np.where(u.toarray().ravel() > 0)
    x = reviews[us][:, ps].T
    y = u.data
    err = 0
    eb = 0
    kf = KFold(len(y), n_folds=4)
    for train, test in kf:
        xc, x1 = movie_norm(x[train])
        reg.fit(xc, y[train] - x1)

        xc, x1 = movie_norm(x[test])
        p = np.array([reg.predict(xi) for xi in xc]).ravel()
        e = (p + x1) - y[test]
        err += np.sum(e * e)
        eb += np.sum((y[train].mean() - y[test]) ** 2)
    return np.sqrt(err / float(len(y))), np.sqrt(eb / float(len(y)))

whole_data = []
for i in range(reviews.shape[0]):
    s = learn_for(i)
    print(s[0] < s[1])
    print(s)
    whole_data.append(s)
