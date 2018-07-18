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
dataset = [frozenset(ds) for ds in dataset]

itemsets = [frozenset([v]) for v in valid]
freqsets = itemsets[:]
for i in range(16):
    print("At iteration {}, number of frequent baskets: {}".format(
        i, len(itemsets)))
    nextsets = []

    tested = set()
    for it in itemsets:
        for v in valid:
            if v not in it:
                # Create a new candidate set by adding v to it
                c = (it | frozenset([v]))

                # Check if we have tested it already:
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


def rules_from_itemset(itemset, dataset, minlift=1.):
    nr_transactions = float(len(dataset))
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

for itemset in freqsets:
    if len(itemset) > 1:
        rules_from_itemset(itemset, dataset, minlift=4.)
