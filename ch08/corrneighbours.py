# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

from __future__ import print_function
import numpy as np
from load_ml100k import get_train_test
from scipy.spatial import distance
from sklearn import metrics

from norm import NormalizePositive

def predict(otrain):
    binary = (otrain > 0)
    norm = NormalizePositive()
    train = norm.fit_transform(otrain.T).T

    proximity = distance.pdist(binary, 'correlation')
    proximity = distance.squareform(proximity)

    neighbors = proximity.argsort(axis=1)
    filled = train.copy()
    for u in range(filled.shape[0]):
        n_u = neighbors[u, 1:]
        t_u = train[n_u].T
        b_u = binary[n_u].T
        for m in range(filled.shape[1]):
            revs = t_u[m]
            brevs = b_u[m]
            revs = revs[brevs]
            if len(revs):
                revs = revs[:len(revs)//2+1]
                filled[u,m] = revs.mean()

    return norm.inverse_transform(filled.T).T

def main(transpose_inputs=False):
    train, test = get_train_test()
    if transpose_inputs:
        train = train.T
        test  = test.T

    predicted = predict(train)
    r2 = metrics.r2_score(test[test > 0], predicted[test > 0])
    print('R2 score (binary neighbours): {:.1%}'.format(r2))

if __name__ == '__main__':
    main()
