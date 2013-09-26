from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
data = load_iris()
features = data['data']
labels = data['target_names'][data['target']]


setosa = (labels == 'setosa')
features = features[~setosa]
labels = labels[~setosa]
virginica = (labels == 'virginica')


best_acc = -1.0
for fi in range(features.shape[1]):
    thresh = features[:,fi].copy()
    thresh.sort()
    for t in thresh:
        pred = (features[:,fi] > t)
        acc = (pred == virginica).mean()
        if acc > best_acc:
            best_acc = acc
            best_fi = fi
            best_t = t
print('Best cut is {0} on feature {1}, which achieves accuracy of {2:.1%}.'.format(best_t,best_fi,best_acc))
