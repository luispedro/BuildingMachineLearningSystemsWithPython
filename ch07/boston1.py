# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

# This script shows an example of simple (ordinary) linear regression

# The first edition of the book NumPy functions only for this operation. See
# the file boston1numpy.py for that version.

import numpy as np
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt

boston = load_boston()
x = boston.data
y = boston.target

# Fitting a model is trivial: call the ``fit`` method in LinearRegression:
lr = LinearRegression()
lr.fit(x, y)

# The instance member `residues_` contains the sum of the squared residues
rmse = np.sqrt(lr.residues_/len(x))
print('RMSE: {}'.format(rmse))

# Plot the prediction versus real:
plt.scatter(lr.predict(x), boston.target)

# Plot a diagonal (for reference):
plt.plot([0, 50], [0, 50], '-', color=(.9,.3,.3), lw=4)
plt.xlabel('predicted')
plt.ylabel('real')
plt.savefig('Figure_07_08.png')
