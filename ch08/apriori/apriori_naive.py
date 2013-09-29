# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

import numpy as np
from collections import defaultdict
from itertools import chain
from gzip import GzipFile
minsupport = 44

dataset = [[int(tok) for tok in line.strip().split()]
           for line in GzipFile('retail.dat.gz')]
dataset = dataset[::20]

counts = defaultdict(int)
for elem in chain(*dataset):
    counts[elem] += 1

valid = set(el for el, c in counts.items() if (c >= minsupport))
dataset = [[el for el in ds if (el in valid)] for ds in dataset]

dataset = [frozenset(ds) for ds in dataset if len(ds) > 1]

itemsets = [frozenset([v]) for v in valid]
allsets = [itemsets]
for i in range(16):
    print(len(itemsets))
    nextsets = []
    for i, ell in enumerate(itemsets):
        for v_ in valid:
            if v_ not in ell:
                c = (ell | set([v_]))
                if sum(1 for d in dataset if d.issuperset(c)) > minsupport:
                    nextsets.append(c)
    allsets.append(nextsets)
    itemsets = nextsets
