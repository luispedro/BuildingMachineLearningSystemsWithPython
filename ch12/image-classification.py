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
from jug import TaskGenerator
from sys import path
path.append('../ch10')


# This is the jug-enabled version of the script ``figure18.py`` in Chapter 10

basedir = '../SimpleImageDataset/'

@TaskGenerator
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
    return mh.features.haralick(im).ravel()

@TaskGenerator
def chist(fname):
    from features import color_histogram
    im = mh.imread(fname)
    return color_histogram(im)


@TaskGenerator
def accuracy(features, labels):
    # We use logistic regression because it is very fast.
    # Feel free to experiment with other classifiers
    scores = cross_validation.cross_val_score(
        LogisticRegression(), features, labels, cv=5)
    return scores.mean()


@TaskGenerator
def stack_features(chists, haralicks):
    return np.hstack([np.atleast_2d(chists), haralicks])

@TaskGenerator
def print_results(scores_base, scores_combined):
    output = open('results.image.txt', 'w')
    output.write('Accuracy (5 fold x-val) with Logistic Regrssion [std features]: {}%\n'.format(
            0.1 * round(1000 * scores_base.mean())))
    output.write('Accuracy (5 fold x-val) with Logistic Regrssion [std features + sobel]: {}%\n'.format(
        0.1 * round(1000 * scores_combined.mean())))
    output.close()


to_array = TaskGenerator(np.array)

haralicks = []
chists = []
labels = []

# Use glob to get all the images
images = glob('{}/*.jpg'.format(basedir))
for fname in sorted(images):
    haralicks.append(features_for(fname))
    chists.append(chist(fname))
    labels.append(fname[:-len('00.jpg')]) # The class is encoded in the filename as xxxx00.jpg

haralicks = to_array(haralicks)
chists = to_array(chists)
labels = to_array(labels)

scores_base = accuracy(haralicks, labels)
haralick_plus_sobel = stack_features(chists, haralicks)
scores_combined  = accuracy(haralick_plus_sobel, labels)

print_results(scores_base, scores_combined)

