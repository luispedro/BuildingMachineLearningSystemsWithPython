from load import load_dataset
import numpy as np
from knn import learn_model, apply_model, accuracy

features,labels = load_dataset('seeds')

def cross_validate(features, labels):
    error = 0.0
    for fold in range(10):
        training = np.ones(len(features), bool)
        training[fold::10] = 0
        testing = ~training
        model = learn_model(1, features[training], labels[training])
        test_error = accuracy(features[testing], labels[testing], model)
        error += test_error

    return error/ 10.0

error = cross_validate(features, labels)
print('Ten fold cross-validated error was {0:.1%}.'.format(error))

features -= features.mean(0)
features /= features.std(0)
error = cross_validate(features, labels)
print('Ten fold cross-validated error after z-scoring was {0:.1%}.'.format(error))

