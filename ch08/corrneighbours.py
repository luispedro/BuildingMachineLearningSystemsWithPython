# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

from __future__ import print_function
from all_correlations import all_correlations
import numpy as np
from scipy import sparse
from load_ml100k import load
reviews = load()


def estimate_user(user, rest):
    bu = user > 0
    br = rest > 0
    ws = all_correlations(bu, br)
    selected = ws.argsort()[-100:]
    estimates = rest[selected].mean(0)
    estimates /= (.1 + br[selected].mean(0))
    return estimates


def train_test(user, rest):
    estimates = estimate_user(user, rest)
    bu = user > 0
    br = rest > 0
    err = estimates[bu] - user[bu]
    null = rest.mean(0)
    null /= (.1 + br.mean(0))
    nerr = null[bu] - user[bu]
    return np.dot(err, err), np.dot(nerr, nerr)


def cross_validate_all():
    err = []
    for i in xrange(reviews.shape[0]):
        err.append(
            train_test(reviews[i], np.delete(reviews, i, 0))
        )
    revs = (reviews > 0).sum(1)
    err = np.array(err)
    rmse = np.sqrt(err / revs[:, None])
    print(np.mean(rmse, 0))
    print(np.mean(rmse[revs > 60], 0))


def all_estimates(reviews):
    reviews = reviews.toarray()
    estimates = np.zeros_like(reviews)
    for i in xrange(reviews.shape[0]):
        estimates[i] = estimate_user(reviews[i], np.delete(reviews, i, 0))
    return estimates
