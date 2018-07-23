# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

import os

from matplotlib import pylab
import numpy as np
from scipy.stats import norm, entropy

from utils import CHART_DIR


def mutual_info(x, y, bins=10):
    counts_xy, bins_x, bins_y = np.histogram2d(x, y, bins=(bins, bins))
    counts_x, bins = np.histogram(x, bins=bins)
    counts_y, bins = np.histogram(y, bins=bins)

    counts_xy += 1
    counts_x += 1
    counts_y += 1
    P_xy = counts_xy / np.sum(counts_xy, dtype=float)
    P_x = counts_x / np.sum(counts_x, dtype=float)
    P_y = counts_y / np.sum(counts_y, dtype=float)

    I_xy = np.sum(P_xy * np.log2(P_xy / (P_x.reshape(-1, 1) * P_y)))

    return I_xy / (entropy(counts_x) + entropy(counts_y))


def plot_entropy():
    pylab.clf()
    pylab.figure(num=None, figsize=(5, 4))

    title = "Entropy $H(X)$"
    pylab.title(title)
    pylab.xlabel("$P(X=$coin will show heads up$)$")
    pylab.ylabel("$H(X)$")

    pylab.xlim(xmin=0, xmax=1.1)
    x = np.arange(0.001, 1, 0.001)
    y = -x * np.log2(x) - (1 - x) * np.log2(1 - x)
    pylab.plot(x, y)
    # pylab.xticks([w*7*24 for w in [0,1,2,3,4]], ['week %i'%(w+1) for w in
    # [0,1,2,3,4]])

    pylab.autoscale(tight=True)
    pylab.grid(True)

    filename = "entropy_demo.png"
    pylab.savefig(os.path.join(CHART_DIR, filename), bbox_inches="tight")


def _plot_mi_func(x, y):

    mi = mutual_info(x, y)
    title = "NI($X_1$, $X_2$) = %.3f" % mi
    pylab.scatter(x, y)
    pylab.title(title)
    pylab.xlabel("$X_1$")
    pylab.ylabel("$X_2$")


def plot_mi_demo():
    np.random.seed(0)  # to reproduce the data later on
    pylab.clf()
    pylab.figure(num=None, figsize=(8, 8))

    x = np.arange(0, 10, 0.2)

    pylab.subplot(221)
    y = 0.5 * x + norm.rvs(1, scale=.01, size=len(x))
    _plot_mi_func(x, y)

    pylab.subplot(222)
    y = 0.5 * x + norm.rvs(1, scale=.1, size=len(x))
    _plot_mi_func(x, y)

    pylab.subplot(223)
    y = 0.5 * x + norm.rvs(1, scale=1, size=len(x))
    _plot_mi_func(x, y)

    pylab.subplot(224)
    y = norm.rvs(1, scale=10, size=len(x))
    _plot_mi_func(x, y)

    pylab.autoscale(tight=True)
    pylab.grid(True)

    filename = "mi_demo_1.png"
    pylab.savefig(os.path.join(CHART_DIR, filename), bbox_inches="tight")

    pylab.clf()
    pylab.figure(num=None, figsize=(8, 8))

    x = np.arange(-5, 5, 0.2)

    pylab.subplot(221)
    y = 0.5 * x ** 2 + norm.rvs(1, scale=.01, size=len(x))
    _plot_mi_func(x, y)

    pylab.subplot(222)
    y = 0.5 * x ** 2 + norm.rvs(1, scale=.1, size=len(x))
    _plot_mi_func(x, y)

    pylab.subplot(223)
    y = 0.5 * x ** 2 + norm.rvs(1, scale=1, size=len(x))
    _plot_mi_func(x, y)

    pylab.subplot(224)
    y = 0.5 * x ** 2 + norm.rvs(1, scale=10, size=len(x))
    _plot_mi_func(x, y)

    pylab.autoscale(tight=True)
    pylab.grid(True)

    filename = "mi_demo_2.png"
    pylab.savefig(os.path.join(CHART_DIR, filename), bbox_inches="tight")

if __name__ == '__main__':
    plot_entropy()
    plot_mi_demo()
