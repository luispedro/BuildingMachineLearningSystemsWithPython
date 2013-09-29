# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

from __future__ import print_function
import numpy as np
from load_ml100k import load
from all_correlations import all_correlations


def nn_movie(ureviews, reviews, uid, mid, k=1):
    X = ureviews
    y = ureviews[mid].copy()
    y -= y.mean()
    y /= (y.std() + 1e-5)
    corrs = np.dot(X, y)
    likes = corrs.argsort()
    likes = likes[::-1]
    c = 0
    pred = 3.
    for ell in likes:
        if ell == mid:
            continue
        if reviews[uid, ell] > 0:
            pred = reviews[uid, ell]
            if c == k:
                return pred
            c += 1
    return pred


def all_estimates(reviews, k=1):
    reviews = reviews.astype(float)
    k -= 1
    nusers, nmovies = reviews.shape
    estimates = np.zeros_like(reviews)
    for u in range(nusers):
        ureviews = np.delete(reviews, u, 0)
        ureviews -= ureviews.mean(0)
        ureviews /= (ureviews.std(0) + 1e-4)
        ureviews = ureviews.T.copy()
        for m in np.where(reviews[u] > 0)[0]:
            estimates[u, m] = nn_movie(ureviews, reviews, u, m, k)
    return estimates

if __name__ == '__main__':
    reviews = load().toarray()
    estimates = all_estimates(reviews)
    error = (estimates - reviews)
    error **= 2
    error = error[reviews > 0]
    print(np.sqrt(error).mean())
