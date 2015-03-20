# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

import numpy as np
import mahotas as mh


def edginess_sobel(image):
    '''Measure the "edginess" of an image

    image should be a 2d numpy array (an image)

    Returns a floating point value which is higher the "edgier" the image is.

    '''
    edges = mh.sobel(image, just_filter=True)
    edges = edges.ravel()
    return np.sqrt(np.dot(edges, edges))

def texture(im):
    '''Compute features for an image

    Parameters
    ----------
    im : ndarray

    Returns
    -------
    fs : ndarray
        1-D array of features
    '''
    im = im.astype(np.uint8)
    return mh.features.haralick(im).ravel()


def chist(im):
    '''Compute color histogram of input image

    Parameters
    ----------
    im : ndarray
        should be an RGB image

    Returns
    -------
    c : ndarray
        1-D array of histogram values
    '''

    # Downsample pixel values:
    im = im // 64

    # We can also implement the following by using np.histogramdd
    # im = im.reshape((-1,3))
    # bins = [np.arange(5), np.arange(5), np.arange(5)]
    # hist = np.histogramdd(im, bins=bins)[0]
    # hist = hist.ravel()

    # Separate RGB channels:
    r,g,b = im.transpose((2,0,1))

    pixels = 1 * r + 4 * g + 16 * b
    hist = np.bincount(pixels.ravel(), minlength=64)
    hist = hist.astype(float)
    return np.log1p(hist)

