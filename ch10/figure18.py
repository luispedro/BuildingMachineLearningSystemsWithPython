# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

import mahotas as mh
from sklearn import cross_validation
from sklearn.linear_model.logistic import LogisticRegression
from mpltools import style
from matplotlib import pyplot as plt
import numpy as np
from glob import glob

basedir = 'AnimTransDistr'


def features_for(images):
    fs = []
    for im in images:
        im = mh.imread(im, as_grey=True).astype(np.uint8)
        fs.append(mh.features.haralick(im).mean(0))
    return np.array(fs)


def features_labels(groups):
    labels = np.zeros(sum(map(len, groups)))
    st = 0
    for i, g in enumerate(groups):
        labels[st:st + len(g)] = i
        st += len(g)
    return np.vstack(groups), labels

classes = [
    'Anims',
    'Cars',
    'Distras',
    'Trans',
]

features = []
labels = []
for ci, cl in enumerate(classes):
    images = glob('{}/{}/*.jpg'.format(basedir, cl))
    features.extend(features_for(images))
    labels.extend([ci for _ in images])

features = np.array(features)
labels = np.array(labels)

scores0 = cross_validation.cross_val_score(
    LogisticRegression(), features, labels, cv=10)
print('Accuracy (5 fold x-val) with Logistic Regrssion [std features]: %s%%' % (
    0.1 * round(1000 * scores0.mean())))

tfeatures = features

from sklearn.cluster import KMeans
from mahotas.features import surf

images = []
labels = []

for ci, cl in enumerate(classes):
    curimages = glob('{}/{}/*.jpg'.format(basedir, cl))
    images.extend(curimages)
    labels.extend([ci for _ in curimages])
labels = np.array(labels)

alldescriptors = []
for im in images:
    im = mh.imread(im, as_grey=1)
    im = im.astype(np.uint8)

    #alldescriptors.append(surf.dense(im, spacing=max(im.shape)//32))
    alldescriptors.append(surf.surf(im, descriptor_only=True))

print('Descriptors done')
k = 256
km = KMeans(k)

concatenated = np.concatenate(alldescriptors)
concatenated = concatenated[::64]
print('k-meaning...')
km.fit(concatenated)
features = []
for d in alldescriptors:
    c = km.predict(d)
    features.append(
        np.array([np.sum(c == i) for i in xrange(k)])
    )
features = np.array(features)
print('predicting...')
scoreSURFlr = cross_validation.cross_val_score(
    LogisticRegression(), features, labels, cv=5).mean()
print('Accuracy (5 fold x-val) with Log. Reg [SURF features]: %s%%' % (
    0.1 * round(1000 * scoreSURFlr.mean())))

print('combined...')
allfeatures = np.hstack([features, tfeatures])
scoreSURFplr = cross_validation.cross_val_score(
    LogisticRegression(), allfeatures, labels, cv=5).mean()

print('Accuracy (5 fold x-val) with Log. Reg [All features]: %s%%' % (
    0.1 * round(1000 * scoreSURFplr.mean())))

style.use('ggplot')
plt.plot([0, 1, 2], 100 *
         np.array([scores0.mean(), scoreSURFlr, scoreSURFplr]), 'k-', lw=8)
plt.plot(
    [0, 1, 2], 100 * np.array([scores0.mean(), scoreSURFlr, scoreSURFplr]),
    'o', mec='#cccccc', mew=12, mfc='white')
plt.xlim(-.5, 2.5)
plt.ylim(scores0.mean() * 90., scoreSURFplr * 110)
plt.xticks([0, 1, 2], ["baseline", "SURF", "combined"])
plt.ylabel('Accuracy (%)')
plt.savefig('../1400OS_10_18+.png')
