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

basedir = 'simple-dataset'


def features_for(im):
    im = mh.imread(im, as_grey=True).astype(np.uint8)
    return mh.features.haralick(im).mean(0)

features = []
sobels = []
labels = []
images = glob('{}/*.jpg'.format(basedir))
for im in images:
    features.append(features_for(im))
    sobels.append(edginess_sobel(mh.imread(im, as_grey=True)))
    labels.append(im[:-len('00.jpg')])

features = np.array(features)
labels = np.array(labels)

scores = cross_validation.cross_val_score(
    LogisticRegression(), features, labels, cv=5)
print('Accuracy (5 fold x-val) with Logistic Regrssion [std features]: {}%'.format(
    0.1 * round(1000 * scores.mean())))

scores = cross_validation.cross_val_score(
    LogisticRegression(), np.hstack([np.atleast_2d(sobels).T, features]), labels, cv=5).mean()
print('Accuracy (5 fold x-val) with Logistic Regrssion [std features + sobel]: {}%'.format(
    0.1 * round(1000 * scores.mean())))
