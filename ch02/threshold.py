import numpy as np
def learn_model(features, labels):
    best_acc = -1.0
    for fi in range(features.shape[1]):
        thresh = features[:,fi].copy()
        thresh.sort()
        for t in thresh:
            pred = (features[:,fi] > t)
            acc = (pred == labels).mean()
            if acc > best_acc:
                best_acc = acc
                best_fi = fi
                best_t = t
    return best_t, best_fi

def apply_model(features, model):
    t, fi = model
    return features[:,fi] > t

def accuracy(features, labels, model):
    preds = apply_model(features, model)
    return np.mean(preds == labels)
