# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

from __future__ import print_function
from sklearn.linear_model import LinearRegression
from load_ml100k import load
import numpy as np
import similar_movie
import usermodel
import corrneighbours

reviews = load()
reg = LinearRegression()
es = np.array([
    usermodel.all_estimates(reviews),
    corrneighbours.all_estimates(reviews),
    similar_movies.all_estimates(reviews),
])

reviews = reviews.toarray()


total_error = 0.0
coefficients = []
for u in xrange(reviews.shape[0]):
    es0 = np.delete(es, u, 1)
    r0 = np.delete(reviews, u, 0)
    X, Y = np.where(r0 > 0)
    X = es[:, X, Y]
    y = r0[r0 > 0]
    reg.fit(X.T, y)
    coefficients.append(reg.coef_)

    r0 = reviews[u]
    X = np.where(r0 > 0)
    p0 = reg.predict(es[:, u, X].squeeze().T)
    err0 = r0[r0 > 0] - p0
    total_error += np.dot(err0, err0)
    print(u)
