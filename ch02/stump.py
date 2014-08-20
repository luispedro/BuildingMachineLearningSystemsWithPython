# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

from sklearn.datasets import load_iris
data = load_iris()
features = data['data']
labels = data['target_names'][data['target']]


is_setosa = (labels == 'setosa')
features = features[~is_setosa]
labels = labels[~is_setosa]
is_virginica = (labels == 'virginica')


# Initialize to a value that is worse than any possible test
best_acc = -1.0

# Loop over all the features
for fi in range(features.shape[1]):
    # Test every possible threshold value for feature fi
    thresh = features[:, fi].copy()

    # Test them in order
    thresh.sort()
    for t in thresh:

        # Generate predictions using t as a threshold
        pred = (features[:, fi] > t)

        # Accuracy is the fraction of predictions that match reality
        acc = (pred == is_virginica).mean()

        # If this is better than previous best, then this is now the new best:

        if acc > best_acc:
            best_acc = acc
            best_fi = fi
            best_t = t
print('Best threshold is {0} on feature {1}, which achieves accuracy of {2:.1%}.'.format(
    best_t, best_fi, best_acc))
