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
    norm = NormalizePositive(axis=1)
    train = norm.fit_transform(otrain)

    dists = distance.pdist(binary, 'correlation')
    dists = distance.squareform(dists)

    neighbors = dists.argsort(axis=1)
    filled = train.copy()
    for u in range(filled.shape[0]):
        # n_u are the neighbors of user
        n_u = neighbors[u, 1:]
        for m in range(filled.shape[1]):
            # This code could be faster using numpy indexing trickery as the
            # cost of readibility (this is left as an exercise to the reader):
            revs = [train[neigh, m]
                    for neigh in n_u
                    if binary[neigh, m]]
            if len(revs):
                n = len(revs)
                n //= 2
                n += 1
                revs = revs[:n]
                filled[u,m] = np.mean(revs)

    return norm.inverse_transform(filled)

def main(transpose_inputs=False):
    train, test = get_train_test(random_state=12)
    if transpose_inputs:
        train = train.T
        test  = test.T

    predicted = predict(train)
    r2 = metrics.r2_score(test[test > 0], predicted[test > 0])
    print('R2 score (binary {} neighbours): {:.1%}'.format(
        ('movie' if transpose_inputs else 'user'),
        r2))

if __name__ == '__main__':
    main()
    main(transpose_inputs=True)
