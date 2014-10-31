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
from features import texture, edginess_sobel

basedir = '../SimpleImageDataset/'


haralicks = []
sobels = []
labels = []

print('This script will test (with cross-validation) classification of the simple 3 class dataset')
print('Computing features...')
# Use glob to get all the images
images = glob('{}/*.jpg'.format(basedir))

# We sort the images to ensure that they are always processed in the same order
# Otherwise, this would introduce some variation just based on the random
# ordering that the filesystem uses
for fname in sorted(images):
    im = mh.imread(fname, as_grey=True)
    haralicks.append(texture(im))
    sobels.append(edginess_sobel(im))

    # Files are named like building00.jpg, scene23.jpg...
    labels.append(fname[:-len('xx.jpg')])

print('Finished computing features.')

haralicks = np.array(haralicks)
sobels = np.array(sobels)
labels = np.array(labels)

# We use logistic regression because it is very fast.
# Feel free to experiment with other classifiers
scores = cross_validation.cross_val_score(
    LogisticRegression(), haralicks, labels, cv=5)
print('Accuracy (5 fold x-val) with Logistic Regression [std features]: {}%'.format(
    0.1 * round(1000 * scores.mean())))

haralick_plus_sobel = np.hstack([np.atleast_2d(sobels).T, haralicks])
scores = cross_validation.cross_val_score(
    LogisticRegression(), haralick_plus_sobel, labels, cv=5).mean()
print('Accuracy (5 fold x-val) with Logistic Regression [std features + sobel]: {}%'.format(
    0.1 * round(1000 * scores.mean())))


# We can try to just use the sobel feature. The result is almost completely
# random.
scores = cross_validation.cross_val_score(
    LogisticRegression(), np.atleast_2d(sobels).T, labels, cv=5).mean()
print('Accuracy (5 fold x-val) with Logistic Regression [only using sobel feature]: {}%'.format(
    0.1 * round(1000 * scores.mean())))

