# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

from __future__ import print_function
from all_correlations import all_correlations
import numpy as np
from load_ml100k import load

def estimate_user(user, rest, num_neighbors=100):
    '''Estimate ratings for user based on the binary rating matrix

    Returns
    -------
    estimates: ndarray
        Returns a rating estimate for each movie
    '''

    # Compute binary matrix correlations:
    bu = user > 0
    br = rest > 0
    ws = all_correlations(bu, br)

    # Select top `num_neighbors`:
    selected = ws.argsort()[-num_neighbors:]

    # Use these to compute estimates:
    estimates = rest[selected].mean(0)
    estimates /= (.1 + br[selected].mean(0))
    return estimates


def train_test(user, rest):
    '''Train & test on a single user

    Returns both the prediction error and the null error
    '''
    estimates = estimate_user(user, rest)
    bu = user > 0
    br = rest > 0
    err = estimates[bu] - user[bu]
    null = rest.mean(0)
    null /= (.1 + br.mean(0))
    nerr = null[bu] - user[bu]
    return np.dot(err, err), np.dot(nerr, nerr)


def all_estimates(reviews):
    estimates = np.zeros_like(reviews)
    for i in range(reviews.shape[0]):
        estimates[i] = estimate_user(reviews[i], np.delete(reviews, i, 0))
    return estimates

def main():
    reviews = load()

    err = []
    for i in range(reviews.shape[0]):
        err.append(
            train_test(reviews[i], np.delete(reviews, i, 0))
        )
    revs = (reviews > 0).sum(1)
    err = np.array(err)
    rmse = np.sqrt(err / revs[:, None])

    rmse_model, rmse_null = np.mean(rmse, 0)

    print("Average of RMSE / Null-model RMSE")
    print("{:.2}\t{:.2} (improvement: {:.1%}".format(rmse_model, rmse_null, (rmse_null-rmse_model)/rmse_null))
    print()

    rmse_model, rmse_null = np.mean(rmse[revs > 60], 0)
    print("Average of RMSE / Null-model RMSE (users with more than 60 reviewed movies)")
    print("{:.2}\t{:.2} (improvement: {:.1%}".format(rmse_model, rmse_null, (rmse_null-rmse_model)/rmse_null))

if __name__ == '__main__':
    main()
