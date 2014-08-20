# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

# This script demonstrates the difference between the training accuracy and
# testing (held-out) accuracy.

import numpy as np
from sklearn.datasets import load_iris
from threshold import fit_model, accuracy

data = load_iris()
features = data['data']
labels = data['target_names'][data['target']]

# We are going to remove the setosa examples as they are too easy:
is_setosa = (labels == 'setosa')
features = features[~is_setosa]
labels = labels[~is_setosa]

# Now we classify virginica vs non-virginica
is_virginica = (labels == 'virginica')

# Split the data in two: testing and training
testing = np.tile([True, False], 50) # testing = [True,False,True,False,True,False...]

# Training is the negation of testing: i.e., datapoints not used for testing,
# will be used for training
training = ~testing

model = fit_model(features[training], is_virginica[training])
train_accuracy = accuracy(features[training], is_virginica[training], model)
test_accuracy = accuracy(features[testing], is_virginica[testing], model)

print('''\
Training accuracy was {0:.1%}.
Testing accuracy was {1:.1%} (N = {2}).
'''.format(train_accuracy, test_accuracy, testing.sum()))
