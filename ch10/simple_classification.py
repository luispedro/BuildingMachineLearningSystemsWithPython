# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

import mahotas as mh
from sklearn import cross_validation
from sklearn.linear_model.logistic import LogisticRegression
import numpy as np
from glob import glob
from edginess import edginess_sobel

basedir = '../SimpleImageDataset/'


def features_for(im):
    '''Compute features for an image

    Parameters
    ----------
    im : str
        filepath for image to process

    Returns
    -------
    fs : ndarray
        1-D array of features
    '''
    im = mh.imread(im, as_grey=True).astype(np.uint8)
    return mh.features.haralick(im).mean(0)

haralicks = []
sobels = []
labels = []

# Use glob to get all the images
images = glob('{}/*.jpg'.format(basedir))
for fname in images:
    haralicks.append(features_for(fname))
    sobels.append(edginess_sobel(mh.imread(fname, as_grey=True)))
    labels.append(fname[:-len('00.jpg')])

haralicks = np.array(haralicks)
sobels = np.array(sobels)
labels = np.array(labels)

# We use logistic regression because it is very fast.
# Feel free to experiment with other classifiers
scores = cross_validation.cross_val_score(
    LogisticRegression(), haralicks, labels, cv=5)
print('Accuracy (5 fold x-val) with Logistic Regrssion [std features]: {}%'.format(
    0.1 * round(1000 * scores.mean())))

haralick_plus_sobel = np.hstack([np.atleast_2d(sobels).T, haralicks])
scores = cross_validation.cross_val_score(
    LogisticRegression(), haralick_plus_sobel, labels, cv=5).mean()
print('Accuracy (5 fold x-val) with Logistic Regrssion [std features + sobel]: {}%'.format(
    0.1 * round(1000 * scores.mean())))
