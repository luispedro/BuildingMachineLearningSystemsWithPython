# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

from load import load_dataset
import numpy as np
from threshold import fit_model, accuracy

features, labels = load_dataset('seeds')

# Turn the labels into a binary array
labels = (labels == 'Canadian')

error = 0.0
for fold in range(10):
    training = np.ones(len(features), bool)

    # numpy magic to make an array with 10% of 0s starting at fold
    training[fold::10] = 0

    # whatever is not training is for testing
    testing = ~training

    model = fit_model(features[training], labels[training])
    test_error = accuracy(features[testing], labels[testing], model)
    error += test_error

error /= 10.0

print('Ten fold cross-validated error was {0:.1%}.'.format(error))
