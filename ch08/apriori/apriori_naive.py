# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

from collections import defaultdict
from itertools import chain
from gzip import GzipFile
minsupport = 80

dataset = [[int(tok) for tok in line.strip().split()]
           for line in GzipFile('retail.dat.gz')]

counts = defaultdict(int)
for elem in chain(*dataset):
    counts[elem] += 1

# Only elements that have at least minsupport should be considered.
valid = set(el for el, c in counts.items() if (c >= minsupport))

# Filter the dataset to contain only valid elements
# (This step is not strictly necessary, but will make the rest of the code
# faster as the itemsets will be smaller):
dataset = [[el for el in ds if (el in valid)] for ds in dataset]

# Convert to frozenset for fast processing
dataset = [frozenset(ds) for ds in dataset if len(ds) > 1]

itemsets = [frozenset([v]) for v in valid]
freqsets = [itemsets]
for i in range(16):
    print("At iteration {}, number of frequent baskets: {}".format(
        i, len(itemsets)))
    nextsets = []
    for it in itemsets:
        for v in valid:
            if v not in it:
                # Create a new candidate set by adding v to it
                c = (it | frozenset([v]))

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
