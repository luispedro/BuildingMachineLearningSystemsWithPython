# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

import mahotas as mh
from mahotas.colors import rgb2grey
import numpy as np

im = mh.imread('lenna.jpg')
im = rgb2grey(im)

salt = np.random.random(im.shape) > .975
pepper = np.random.random(im.shape) > .975

im = np.maximum(salt * 170, mh.stretch(im))
im = np.minimum(pepper * 30 + im * (~pepper), im)

mh.imsave('../1400OS_10_13+.jpg', im.astype(np.uint8))
