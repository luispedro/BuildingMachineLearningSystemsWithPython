import numpy as np # NOT IN BOOK
from matplotlib import pyplot as plt # NOT IN BOOK

def load():
    import numpy as np
    from scipy import sparse

    data = np.loadtxt('data/ml-100k/u.data')
    ij = data[:, :2]
    ij -= 1  # original data is in 1-based system
    values = data[:, 2]
    reviews = sparse.csc_matrix((values, ij.T)).astype(float)
    return reviews.toarray()
reviews = load()
U,M = np.where(reviews)
import random
test_idxs = np.array(random.sample(range(len(U)), len(U)//10))

train = reviews.copy()
train[U[test_idxs], M[test_idxs]] = 0

test = np.zeros_like(reviews)
test[U[test_idxs], M[test_idxs]] = reviews[U[test_idxs], M[test_idxs]]

class NormalizePositive(object):
    def __init__(self, axis=0):
        self.axis = axis

    def fit(self, features, y=None):
        if self.axis == 1:
            features = features.T
          #  count features that are greater than zero in axis 0:
        binary = (features > 0)

        count0 = binary.sum(axis=0)

         # to avoid division by zero, set zero counts to one:
        count0[count0 == 0] = 1.

         # computing the mean is easy:
        self.mean = features.sum(axis=0)/count0

        # only consider differences where binary is True:
        diff = (features - self.mean) * binary
        diff **= 2
        # regularize the estimate of std by adding 0.1
        self.std = np.sqrt(0.1 + diff.sum(axis=0)/count0)
        return self


    def transform(self, features):
        if self.axis == 1:
          features = features.T
        binary = (features > 0)
        features = features - self.mean
        features /= self.std
        features *= binary
        if self.axis == 1:
          features = features.T
        return features

    def inverse_transform(self, features, copy=True):
        if copy:
            features = features.copy()
        if self.axis == 1:
          features = features.T
        features *= self.std
        features += self.mean
        if self.axis == 1:
          features = features.T
        return features

    def fit_transform(self, features):
        return self.fit(features).transform(features)


norm = NormalizePositive(axis=1)
binary = (train > 0)
train = norm.fit_transform(train)
# plot just 200x200 area for space reasons
plt.imshow(binary[:200, :200], interpolation='nearest')

from scipy.spatial import distance
# compute all pair-wise distances:
dists = distance.pdist(binary, 'correlation')
# Convert to square form, so that dists[i,j]
# is distance between binary[i] and binary[j]:
dists = distance.squareform(dists)
neighbors = dists.argsort(axis=1)

# We are going to fill this matrix with results
filled = train.copy()
for u in range(filled.shape[0]):
    # n_u is neighbors of user
    n_u = neighbors[u, 1:]
    for m in range(filled.shape[1]):
        # get relevant reviews in order!
        revs = [train[neigh, m]
                   for neigh in n_u
                        if binary    [neigh, m]]
        if len(revs):
            # n is the number of reviews for this movie
            n = len(revs)
            # take half of the reviews plus one into consideration:
            n //= 2
            n += 1
            revs = revs[:n]
            filled[u,m] = np.mean(revs)

predicted = norm.inverse_transform(filled)
from sklearn import metrics
r2 = metrics.r2_score(test[test > 0], predicted[test > 0])
print('R2 score (binary neighbors): {:.1%}'.format(r2))

reviews = reviews.T
# use same code as before 
r2 = metrics.r2_score(test[test > 0], predicted[test > 0])
print('R2 score (binary movie neighbors): {:.1%}'.format(r2))


from sklearn.linear_model import ElasticNetCV # NOT IN BOOK

reg = ElasticNetCV(alphas=[
                       0.0125, 0.025, 0.05, .125, .25, .5, 1., 2., 4.])
filled = train.copy()
# iterate over all users:
for u in range(train.shape[0]):
    curtrain = np.delete(train, u, axis=0)
    bu = binary[u]
    reg.fit(curtrain[:,bu].T, train[u, bu])
    filled[u, ~bu] = reg.predict(curtrain[:,~bu].T)
predicted = norm.inverse_transform(filled)
r2 = metrics.r2_score(test[test > 0], predicted[test > 0])
print('R2 score (user regression): {:.1%}'.format(r2))


# SHOPPING BASKET ANALYSIS
# This is the slow version of the code, which will take a long time to
# complete.


from collections import defaultdict
from itertools import chain

# File is downloaded as a compressed file
import gzip
# file format is a line per transaction
# of the form '12 34 342 5...'
dataset = [[int(tok) for tok in line.strip().split()]
       for line in gzip.open('data/retail.dat.gz')]
dataset = [set(d) for d in dataset]
# count how often each product was purchased:
counts = defaultdict(int)
for elem in chain(*dataset):
    counts[elem] += 1

minsupport = 80
valid = set(k for k,v in counts.items() if (v >= minsupport))
itemsets = [frozenset([v]) for v in valid]
freqsets = []
for i in range(16):
    nextsets = []
    tested = set()
    for it in itemsets:
        for v in valid:
           if v not in it:
               # Create a new candidate set by adding v to it
               c = (it | frozenset([v]))
               # check If we have tested it already
               if c in tested:
                    continue
               tested.add(c)

               # Count support by looping over dataset
               # This step is slow.
               # Check `apriori.py` for a better implementation.
               support_c = sum(1 for d in dataset if d.issuperset(c))
               if support_c > minsupport:
                    nextsets.append(c)
    freqsets.extend(nextsets)
    itemsets = nextsets
    if not len(itemsets):
        break
print("Finished!")


minlift = 5.0
nr_transactions = float(len(dataset))
for itemset in freqsets:
    for item in itemset:
        consequent = frozenset([item])
        antecedent = itemset-consequent
        base = 0.0
        # acount: antecedent count
        acount = 0.0

        # ccount : consequent count
        ccount = 0.0
        for d in dataset:
          if item in d: base += 1
          if d.issuperset(itemset): ccount += 1
          if d.issuperset(antecedent): acount += 1
        base /= nr_transactions
        p_y_given_x = ccount/acount
        lift = p_y_given_x / base
        if lift > minlift:
            print('Rule {0} ->  {1} has lift {2}'
                  .format(antecedent, consequent,lift))
