# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

from sklearn.datasets import load_iris

data = load_iris()
features = data['data']
target = data['target']
target_names = data['target_names']
labels = target_names[target]
plength = features[:, 2]

# To use numpy operations to get setosa features,
# we build a boolean array
is_setosa = (labels == 'setosa')

max_setosa = plength[is_setosa].max()
min_non_setosa = plength[~is_setosa].min()

print('Maximum of setosa: {0}.'.format(max_setosa))
print('Minimum of others: {0}.'.format(min_non_setosa))
