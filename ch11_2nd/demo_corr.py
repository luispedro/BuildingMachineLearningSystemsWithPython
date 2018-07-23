# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

import os

from matplotlib import pylab
import numpy as np
import scipy
from scipy.stats import norm, pearsonr

from utils import CHART_DIR


def _plot_correlation_func(x, y):

    r, p = pearsonr(x, y)
    title = "Cor($X_1$, $X_2$) = %.3f" % r
    pylab.scatter(x, y)
    pylab.title(title)
    pylab.xlabel("$X_1$")
    pylab.ylabel("$X_2$")

    f1 = scipy.poly1d(scipy.polyfit(x, y, 1))
    pylab.plot(x, f1(x), "r--", linewidth=2)
    # pylab.xticks([w*7*24 for w in [0,1,2,3,4]], ['week %i'%(w+1) for w in
    # [0,1,2,3,4]])


def plot_correlation_demo():
    np.random.seed(0)  # to reproduce the data later on
    pylab.clf()
    pylab.figure(num=None, figsize=(8, 8))

    x = np.arange(0, 10, 0.2)

    pylab.subplot(221)
    y = 0.5 * x + norm.rvs(1, scale=.01, size=len(x))
    _plot_correlation_func(x, y)

    pylab.subplot(222)
    y = 0.5 * x + norm.rvs(1, scale=.1, size=len(x))
    _plot_correlation_func(x, y)

    pylab.subplot(223)
    y = 0.5 * x + norm.rvs(1, scale=1, size=len(x))
    _plot_correlation_func(x, y)

    pylab.subplot(224)
    y = norm.rvs(1, scale=10, size=len(x))
    _plot_correlation_func(x, y)

    pylab.autoscale(tight=True)
    pylab.grid(True)

    filename = "corr_demo_1.png"
    pylab.savefig(os.path.join(CHART_DIR, filename), bbox_inches="tight")

    pylab.clf()
    pylab.figure(num=None, figsize=(8, 8))

    x = np.arange(-5, 5, 0.2)

    pylab.subplot(221)
    y = 0.5 * x ** 2 + norm.rvs(1, scale=.01, size=len(x))
    _plot_correlation_func(x, y)

    pylab.subplot(222)
    y = 0.5 * x ** 2 + norm.rvs(1, scale=.1, size=len(x))
    _plot_correlation_func(x, y)

    pylab.subplot(223)
    y = 0.5 * x ** 2 + norm.rvs(1, scale=1, size=len(x))
    _plot_correlation_func(x, y)

    pylab.subplot(224)
    y = 0.5 * x ** 2 + norm.rvs(1, scale=10, size=len(x))
    _plot_correlation_func(x, y)

    pylab.autoscale(tight=True)
    pylab.grid(True)

    filename = "corr_demo_2.png"
    pylab.savefig(os.path.join(CHART_DIR, filename), bbox_inches="tight")

if __name__ == '__main__':
    plot_correlation_demo()
