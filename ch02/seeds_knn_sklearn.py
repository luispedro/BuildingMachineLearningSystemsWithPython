# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

# Basic imports
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
print(correct/n)

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
print(means)

# The function cross_val_score does the same thing as the loop above with a
# single function call

from sklearn.cross_validation import cross_val_score
print(cross_val_score(classifier, features, labels))
