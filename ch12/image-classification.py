# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

import mahotas as mh
import numpy as np
from glob import glob
from jug import TaskGenerator

# We need to use the `features` module from chapter 10.
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
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn import cross_validation
    # We use logistic regression because it is very fast.
    # Feel free to experiment with other classifiers
    clf = Pipeline([('preproc', StandardScaler()),
                ('classifier', LogisticRegression())])
    cv = cross_validation.LeaveOneOut(len(features))
    scores = cross_validation.cross_val_score(
        clf, features, labels, cv=cv)
    return scores.mean()


@TaskGenerator
def stack_features(chists, haralicks):
    return np.hstack([chists, haralicks])

@TaskGenerator
def print_results(scores):
    with open('results.image.txt', 'w') as output:
        for k in scores:
            output.write('Accuracy (LOO x-val) with Logistic Regression [{}]: {:.1%}\n'.format(
                k, scores[k].mean()))


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
scores_chist = accuracy(chists, labels)

combined = stack_features(chists, haralicks)
scores_combined  = accuracy(combined, labels)

print_results({
        'base': scores_base,
        'chists': scores_chist,
        'combined' : scores_combined,
        })

