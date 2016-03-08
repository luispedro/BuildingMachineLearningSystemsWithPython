# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

import mahotas as mh
import numpy as np
from glob import glob

from features import texture, chist
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

basedir = '../SimpleImageDataset/'


haralicks = []
labels = []
chists = []

print('This script will test (with cross-validation) classification of the simple 3 class dataset')
print('Computing features...')
# Use glob to get all the images
images = glob('{}/*.jpg'.format(basedir))

# We sort the images to ensure that they are always processed in the same order
# Otherwise, this would introduce some variation just based on the random
# ordering that the filesystem uses
for fname in sorted(images):
    imc = mh.imread(fname)
    haralicks.append(texture(mh.colors.rgb2grey(imc)))
    chists.append(chist(imc))

    # Files are named like building00.jpg, scene23.jpg...
    labels.append(fname[:-len('xx.jpg')])

print('Finished computing features.')

haralicks = np.array(haralicks)
labels = np.array(labels)
chists = np.array(chists)

haralick_plus_chists = np.hstack([chists, haralicks])


# We use Logistic Regression because it achieves high accuracy on small(ish) datasets
# Feel free to experiment with other classifiers
clf = Pipeline([('preproc', StandardScaler()),
                ('classifier', LogisticRegression())])

from sklearn import cross_validation
cv = cross_validation.LeaveOneOut(len(images))
scores = cross_validation.cross_val_score(
    clf, haralicks, labels, cv=cv)
print('Accuracy (Leave-one-out) with Logistic Regression [haralick features]: {:.1%}'.format(
    scores.mean()))

scores = cross_validation.cross_val_score(
    clf, chists, labels, cv=cv)
print('Accuracy (Leave-one-out) with Logistic Regression [color histograms]: {:.1%}'.format(
    scores.mean()))

scores = cross_validation.cross_val_score(
    clf, haralick_plus_chists, labels, cv=cv)
print('Accuracy (Leave-one-out) with Logistic Regression [texture features + color histograms]: {:.1%}'.format(
    scores.mean()))

