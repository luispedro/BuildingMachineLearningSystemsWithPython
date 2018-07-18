# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

import os

import numpy as np
from matplotlib import pylab
from mpl_toolkits.mplot3d import Axes3D

from sklearn import linear_model, manifold, decomposition, datasets
logistic = linear_model.LogisticRegression()

from utils import CHART_DIR

np.random.seed(3)

# all examples will have three classes in this file
colors = ['r', 'g', 'b']
markers = ['o', 6, '*']


def plot_demo_1():
    X = np.c_[np.ones(5), 2 * np.ones(5), 10 * np.ones(5)].T
    y = np.array([0, 1, 2])

    fig = pylab.figure(figsize=(10, 4))

    ax = fig.add_subplot(121, projection='3d')
    ax.set_axis_bgcolor('white')

    mds = manifold.MDS(n_components=3)
    Xtrans = mds.fit_transform(X)

    for cl, color, marker in zip(np.unique(y), colors, markers):
        ax.scatter(
            Xtrans[y == cl][:, 0], Xtrans[y == cl][:, 1], Xtrans[y == cl][:, 2], c=color, marker=marker, edgecolor='black')
    pylab.title("MDS on example data set in 3 dimensions")
    ax.view_init(10, -15)

    mds = manifold.MDS(n_components=2)
    Xtrans = mds.fit_transform(X)

    ax = fig.add_subplot(122)
    for cl, color, marker in zip(np.unique(y), colors, markers):
        ax.scatter(
            Xtrans[y == cl][:, 0], Xtrans[y == cl][:, 1], c=color, marker=marker, edgecolor='black')
    pylab.title("MDS on example data set in 2 dimensions")

    filename = "mds_demo_1.png"
    pylab.savefig(os.path.join(CHART_DIR, filename), bbox_inches="tight")


def plot_iris_mds():

    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # MDS

    fig = pylab.figure(figsize=(10, 4))

    ax = fig.add_subplot(121, projection='3d')
    ax.set_axis_bgcolor('white')

    mds = manifold.MDS(n_components=3)
    Xtrans = mds.fit_transform(X)

    for cl, color, marker in zip(np.unique(y), colors, markers):
        ax.scatter(
            Xtrans[y == cl][:, 0], Xtrans[y == cl][:, 1], Xtrans[y == cl][:, 2], c=color, marker=marker, edgecolor='black')
    pylab.title("MDS on Iris data set in 3 dimensions")
    ax.view_init(10, -15)

    mds = manifold.MDS(n_components=2)
    Xtrans = mds.fit_transform(X)

    ax = fig.add_subplot(122)
    for cl, color, marker in zip(np.unique(y), colors, markers):
        ax.scatter(
            Xtrans[y == cl][:, 0], Xtrans[y == cl][:, 1], c=color, marker=marker, edgecolor='black')
    pylab.title("MDS on Iris data set in 2 dimensions")

    filename = "mds_demo_iris.png"
    pylab.savefig(os.path.join(CHART_DIR, filename), bbox_inches="tight")

    # PCA

    fig = pylab.figure(figsize=(10, 4))

    ax = fig.add_subplot(121, projection='3d')
    ax.set_axis_bgcolor('white')

    pca = decomposition.PCA(n_components=3)
    Xtrans = pca.fit(X).transform(X)

    for cl, color, marker in zip(np.unique(y), colors, markers):
        ax.scatter(
            Xtrans[y == cl][:, 0], Xtrans[y == cl][:, 1], Xtrans[y == cl][:, 2], c=color, marker=marker, edgecolor='black')
    pylab.title("PCA on Iris data set in 3 dimensions")
    ax.view_init(50, -35)

    pca = decomposition.PCA(n_components=2)
    Xtrans = pca.fit_transform(X)

    ax = fig.add_subplot(122)
    for cl, color, marker in zip(np.unique(y), colors, markers):
        ax.scatter(
            Xtrans[y == cl][:, 0], Xtrans[y == cl][:, 1], c=color, marker=marker, edgecolor='black')
    pylab.title("PCA on Iris data set in 2 dimensions")

    filename = "pca_demo_iris.png"
    pylab.savefig(os.path.join(CHART_DIR, filename), bbox_inches="tight")


if __name__ == '__main__':
    plot_demo_1()
    plot_iris_mds()
