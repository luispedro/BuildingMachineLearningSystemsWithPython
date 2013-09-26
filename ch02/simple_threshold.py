import numpy as np
from sklearn.datasets import load_iris

data = load_iris()
features = data['data']
target = data['target']
target_names = data['target_names']
labels = target_names[target]

plength = features[:,2]
is_setosa = (labels == 'setosa')
print('Maximum of setosa: {0}.'.format(plength[is_setosa].max()))
print('Minimum of others: {0}.'.format(plength[~is_setosa].min()))
