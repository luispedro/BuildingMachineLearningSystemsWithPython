# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

# Basic imports
from __future__ import print_function
import numpy as np
from matplotlib import pyplot as plt
from load import load_dataset


from sklearn.neighbors import KNeighborsClassifier

from sklearn.cross_validation import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


features, labels = load_dataset('seeds')

# Values of k to consider: all in 1 .. 160
ks = np.arange(1,161)

# We build a classifier object here with the default number of neighbors
# (It happens to be 5, but it does not matter as we will be changing it below
classifier = KNeighborsClassifier()
classifier = Pipeline([('norm', StandardScaler()), ('knn', classifier)])

# accuracies will hold our results
accuracies = []
for k in ks:
    # set the classifier parameter
    classifier.set_params(knn__n_neighbors=k)
    crossed = cross_val_score(classifier, features, labels)

    # Save only the average
    accuracies.append(crossed.mean())

accuracies = np.array(accuracies)

# Scale the accuracies by 100 to plot as a percentage instead of as a fraction
plt.plot(ks, accuracies*100)
plt.xlabel('Value for k (nr. of neighbors)')
plt.ylabel('Accuracy (%)')
plt.savefig('figure6.png')
