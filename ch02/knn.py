# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

import numpy as np

# This function was called ``learn_model`` in the first edition
def fit_model(k, features, labels):
    '''Learn a k-nn model'''
    # There is no model in k-nn, just a copy of the inputs
    return k, features.copy(), labels.copy()


def plurality(xs):
    '''Find the most common element in a collection'''
    from collections import defaultdict
    counts = defaultdict(int)
    for x in xs:
        counts[x] += 1
    maxv = max(counts.values())
    for k, v in counts.items():
        if v == maxv:
            return k

# This function was called ``apply_model`` in the first edition
def predict(model, features):
    '''Apply k-nn model'''
    k, train_feats, labels = model
    results = []
    for f in features:
        label_dist = []
        # Compute all distances:
        for t, ell in zip(train_feats, labels):
            label_dist.append((np.linalg.norm(f - t), ell))
        label_dist.sort(key=lambda d_ell: d_ell[0])
        label_dist = label_dist[:k]
        results.append(plurality([ell for _, ell in label_dist]))
    return np.array(results)


def accuracy(features, labels, model):
    preds = predict(model, features)
    return np.mean(preds == labels)
