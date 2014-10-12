# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

import mahotas as mh
from sklearn import cross_validation
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV


import numpy as np
from glob import glob
from features import texture, edginess_sobel, color_histogram

basedir = '../SimpleImageDataset/'


haralicks = []
sobels = []
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
    im = mh.imread(fname, as_grey=True)
    imc = mh.imread(fname)
    haralicks.append(texture(im))
    sobels.append(edginess_sobel(im))
    chists.append(color_histogram(imc))

    # Files are named like building00.jpg, scene23.jpg...
    labels.append(fname[:-len('xx.jpg')])

print('Finished computing features.')

haralicks = np.array(haralicks)
sobels = np.array(sobels)
labels = np.array(labels)
chists = np.array(chists)

haralick_plus_sobel = np.hstack([np.atleast_2d(sobels).T, haralicks])
haralick_plus_chists = np.hstack([chists, haralicks])
haralick_plus_chists_plus_sobel = np.hstack([chists, haralicks, np.atleast_2d(sobels).T])

cv=cross_validation.LeaveOneOut(len(images))

# We use SVM because it achieves high accuracy on small(ish) datasets
# Feel free to experiment with other classifiers
C_range = 10.0 ** np.arange(-2, 9)
gamma_range = 10.0 ** np.arange(-5, 4)
param_grid = dict(gamma=gamma_range, C=C_range)
grid = GridSearchCV(SVC(), param_grid=param_grid)

p = Pipeline([('preproc', StandardScaler()),
                ('classifier', grid)])

scores = cross_validation.cross_val_score(
    p, haralicks, labels, cv=cv)
print('Accuracy (5 fold x-val) with Logistic Regrssion [haralick features]: {}%'.format(
    0.1 * round(1000 * scores.mean())))

scores = cross_validation.cross_val_score(
    p, chists, labels, cv=cv)
print('Accuracy (5 fold x-val) with Logistic Regrssion [color histograms]: {}%'.format(
    0.1 * round(1000 * scores.mean())))

scores = cross_validation.cross_val_score(
    p, haralick_plus_chists, labels, cv=cv)
print('Accuracy (5 fold x-val) with Logistic Regrssion [texture features + chists]: {}%'.format(
    0.1 * round(1000 * scores.mean())))

haralick_plus_sobel = np.hstack([np.atleast_2d(sobels).T, haralicks])
scores = cross_validation.cross_val_score(
    p, haralick_plus_sobel, labels, cv=cv)
print('Accuracy (5 fold x-val) with Logistic Regrssion [texture features + sobel]: {}%'.format(
    0.1 * round(1000 * scores.mean())))

haralick_plus_chists_plus_sobel = np.hstack([np.atleast_2d(sobels).T, haralicks])
scores = cross_validation.cross_val_score(
    p, haralick_plus_chists_plus_sobel, labels, cv=cv)
print('Accuracy (5 fold x-val) with Logistic Regrssion [texture features + color histograms + sobel]: {}%'.format(
    0.1 * round(1000 * scores.mean())))


# We can try to just use the sobel feature. The result is almost completely
# random.
scores = cross_validation.cross_val_score(
    p, np.atleast_2d(sobels).T, labels, cv=cv).mean()
print('Accuracy (5 fold x-val) with Logistic Regrssion [only using sobel feature]: {}%'.format(
    0.1 * round(1000 * scores.mean())))

