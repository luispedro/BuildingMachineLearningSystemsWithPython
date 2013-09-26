from load import load_dataset
import numpy as np
from threshold import learn_model, apply_model, accuracy

features,labels = load_dataset('seeds')
labels = labels == 'Canadian'

error = 0.0
for fold in range(10):
    training = np.ones(len(features), bool)
    training[fold::10] = 0
    testing = ~training
    model = learn_model(features[training], labels[training])
    test_error = accuracy(features[testing], labels[testing], model)
    error += test_error

error /= 10.0

print('Ten fold cross-validated error was {0:.1%}.'.format(error))

