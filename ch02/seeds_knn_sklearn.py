# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

# Basic imports
from __future__ import print_function
import numpy as np
from load import load_dataset


# Import sklearn implementation of KNN
from sklearn.neighbors import KNeighborsClassifier

features, labels = load_dataset('seeds')
classifier = KNeighborsClassifier(n_neighbors=4)


n = len(features)
correct = 0.0
for ei in range(n):
    training = np.ones(n, bool)
    training[ei] = 0
    testing = ~training
    classifier.fit(features[training], labels[training])
    pred = classifier.predict(features[ei])
    correct += (pred == labels[ei])
print('Result of leave-one-out: {}'.format(correct/n))

# Import KFold object
from sklearn.cross_validation import KFold

# means will hold the mean for each fold
means = []

# kf is a generator of pairs (training,testing) so that each iteration
# implements a separate fold.
kf = KFold(len(features), n_folds=3, shuffle=True)
for training,testing in kf:
    # We learn a model for this fold with `fit` and then apply it to the
    # testing data with `predict`:
    classifier.fit(features[training], labels[training])
    prediction = classifier.predict(features[testing])

    # np.mean on an array of booleans returns the fraction of correct decisions
    # for this fold:
    curmean = np.mean(prediction == labels[testing])
    means.append(curmean)
print('Result of cross-validation using KFold: {}'.format(means))

# The function cross_val_score does the same thing as the loop above with a
# single function call

from sklearn.cross_validation import cross_val_score
crossed = cross_val_score(classifier, features, labels)
print('Result of cross-validation using cross_val_score: {}'.format(crossed))

# The results above use the features as is, which we learned was not optimal
# except if the features happen to all be in the same scale. We can pre-scale
# the features as explained in the main text:

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
classifier = Pipeline([('norm', StandardScaler()), ('knn', classifier)])
crossed = cross_val_score(classifier, features, labels)
print('Result with prescaling: {}'.format(crossed))


# Now, generate & print a cross-validated confusion matrix for the same result
from sklearn.metrics import confusion_matrix
names = list(set(labels))
labels = np.array([names.index(ell) for ell in labels])
preds = labels.copy()
preds[:] = -1
for train, test in kf:
    classifier.fit(features[train], labels[train])
    preds[test] = classifier.predict(features[test])

cmat = confusion_matrix(labels, preds)
print()
print('Confusion matrix: [rows represent true outcome, columns predicted outcome]')
print(cmat)

# The explicit float() conversion is necessary in Python 2
# (Otherwise, result is rounded to 0)
acc = cmat.trace()/float(cmat.sum())
print('Accuracy: {0:.1%}'.format(acc))

