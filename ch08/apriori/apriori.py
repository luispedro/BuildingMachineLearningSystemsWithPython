# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

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
    pointers = dict([(k, frozenset(v)) for k, v in pointers.items()])
    baskets = dict([(k, frozenset(v)) for k, v in baskets.items()])

    valid = set(list(el)[0]
                for el, c in baskets.items() if (len(c) >= minsupport))
    dataset = [[el for el in ds if (el in valid)] for ds in dataset]
    dataset = [ds for ds in dataset if len(ds) > 1]
    dataset = map(frozenset, dataset)

    itemsets = [frozenset([v]) for v in valid]
    freqsets = []
    for i in range(maxsize - 1):
        print(len(itemsets))
        newsets = []
        for i, ell in enumerate(itemsets):
            ccounts = baskets[ell]
            for v_, pv in pointers.items():
                if v_ not in ell:
                    csup = (ccounts & pv)
                    if len(csup) >= minsupport:
                        new = frozenset(ell | set([v_]))
                        if new not in baskets:
                            newsets.append(new)
                            baskets[new] = csup
        freqsets.extend(itemsets)
        itemsets = newsets
    return freqsets, baskets


def association_rules(dataset, freqsets, baskets, minlift):
    '''
    for (antecendent, consequent, base, py_x, lift) in association_rules(dataset, freqsets, baskets, minlift):
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
    '''
    nr_transactions = float(len(dataset))
    freqsets = [f for f in freqsets if len(f) > 1]
    for fset in freqsets:
        for f in fset:
            consequent = frozenset([f])
            antecendent = fset - consequent
            base = len(baskets[consequent]) / nr_transactions
            py_x = len(baskets[fset]) / float(len(baskets[antecendent]))
            lift = py_x / base
            if lift > minlift:
                yield (antecendent, consequent, base, py_x, lift)
