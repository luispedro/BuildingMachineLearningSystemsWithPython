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
    return mh.features.haralick(im).mean(0)


