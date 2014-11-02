# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

from collections import namedtuple


def apriori(dataset, minsupport, maxsize):
    '''
    freqsets, baskets = apriori(dataset, minsupport, maxsize)

    Parameters
    ----------
    dataset : sequence of sequences
        input dataset
    minsupport : int
        Minimal support for frequent items
    maxsize : int
        Maximal size of frequent items to return

    Returns
    -------
    freqsets : sequence of sequences
    baskets : dictionary
    '''
    from collections import defaultdict

    baskets = defaultdict(list)

    pointers = defaultdict(list)
    for i, ds in enumerate(dataset):
        for ell in ds:
            pointers[ell].append(i)
            baskets[frozenset([ell])].append(i)

    # Convert pointer items to frozensets to speed up operations later
    new_pointers = dict()
    for k in pointers:
        if len(pointers[k]) >= minsupport:
            new_pointers[k] = frozenset(pointers[k])
    pointers = new_pointers
    for k in baskets:
        baskets[k] = frozenset(baskets[k])


    # Valid are all elements whose support is >= minsupport
    valid = set()
    for el, c in baskets.items():
        if len(c) >= minsupport:
            valid.update(el)

    # Itemsets at first iteration are simply all singleton with valid elements:
    itemsets = [frozenset([v]) for v in valid]
    freqsets = []
    for i in range(maxsize - 1):
        newsets = []
        for it in itemsets:
            ccounts = baskets[it]

            for v, pv in pointers.items():
                if v not in it:
                    csup = (ccounts & pv)
                    if len(csup) >= minsupport:
                        new = frozenset(it | frozenset([v]))
                        if new not in baskets:
                            newsets.append(new)
                            baskets[new] = csup
        freqsets.extend(itemsets)
        itemsets = newsets
    return freqsets, baskets


# A namedtuple to collect all values that may be interesting
AssociationRule = namedtuple('AssociationRule', ['antecendent', 'consequent', 'base', 'py_x', 'lift'])

def association_rules(dataset, freqsets, baskets, minlift):
    '''
    for assoc_rule in association_rules(dataset, freqsets, baskets, minlift):
        ...

    This function takes the returns from ``apriori``.

    Parameters
    ----------
    dataset : sequence of sequences
        input dataset
    freqsets : sequence of sequences
    baskets : dictionary
    minlift : int
        minimal lift of yielded rules

    Returns
    -------
    assoc_rule : sequence of AssociationRule objects
    '''
    nr_transactions = float(len(dataset))
    freqsets = [f for f in freqsets if len(f) > 1]
    for fset in freqsets:
        for f in fset:
            consequent = frozenset([f])
            antecendent = fset - consequent
            py_x = len(baskets[fset]) / float(len(baskets[antecendent]))
            base = len(baskets[consequent]) / nr_transactions
            lift = py_x / base
            if lift > minlift:
                yield AssociationRule(antecendent, consequent, base, py_x, lift)

