import numpy as np
from scipy import sparse
def load():
    data = np.array([[int(t) for t in line.split('\t')[:3]] for line in open('data/ml-100k/u.data')])
    ij = data[:,:2]
    ij -= 1 # original data is in 1-based system
    values = data[:,2]
    reviews = sparse.csc_matrix((values,ij.T)).astype(float)
    return reviews

