import numpy as np
def learn_model(k, features, labels):
    return k, features.copy(),labels.copy()

def plurality(xs):
    from collections import defaultdict
    counts = defaultdict(int)
    for x in xs:
        counts[x] += 1
    maxv = max(counts.values())
    for k,v in counts.items():
        if v == maxv:
            return k

def apply_model(features, model):
    k, train_feats, labels = model
    results = []
    for f in features:
        label_dist = []
        for t,ell in zip(train_feats, labels):
            label_dist.append( (np.linalg.norm(f-t), ell) )
        label_dist.sort(key=lambda d_ell: d_ell[0])
        label_dist = label_dist[:k]
        results.append(plurality([ell for _,ell in label_dist]))
    return np.array(results)

def accuracy(features, labels, model):
    preds = apply_model(features, model)
    return np.mean(preds == labels)
