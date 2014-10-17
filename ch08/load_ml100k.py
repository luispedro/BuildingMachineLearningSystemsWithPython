# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

def load():
    '''Load ML-100k data

    Returns the review matrix as a numpy array'''
    import numpy as np
    from scipy import sparse

    # The input is in the form of a CSC sparse matrix, so it's a natural fit to
    # load the data, but we then convert to a more traditional array before
    # returning
    data = np.loadtxt('data/ml-100k/u.data')
    ij = data[:, :2]
    ij -= 1  # original data is in 1-based system
    values = data[:, 2]
    reviews = sparse.csc_matrix((values, ij.T)).astype(float)
    return reviews.toarray()

def get_train_test():
    import numpy as np
    import random
    reviews = load()
    U,M = np.where(reviews)
    test_idxs = np.array(random.sample(range(len(U)), len(U)//10))
    train = reviews.copy()
    train[U[test_idxs], M[test_idxs]] = 0

    test = np.zeros_like(reviews)
    test[U[test_idxs], M[test_idxs]] = reviews[U[test_idxs], M[test_idxs]]

    valid_U = train.any(0)
    valid_M = train.any(1)
    train = train[valid_M].T[valid_U].T
    test  = test [valid_M].T[valid_U].T
    return train, test

