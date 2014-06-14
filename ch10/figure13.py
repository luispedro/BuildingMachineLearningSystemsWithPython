# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

import mahotas as mh
from mahotas.colors import rgb2grey
import numpy as np

# Adds a little salt-n-pepper noise to an image

im = mh.demos.load('lena')
im = rgb2grey(im)

# Salt & pepper arrays
salt = np.random.random(im.shape) > .975
pepper = np.random.random(im.shape) > .975

# salt is 170 & pepper is 30
# Some playing around showed that setting these to more extreme values looks
# very artificial. These look nicer

im = np.maximum(salt * 170, mh.stretch(im))
im = np.minimum(pepper * 30 + im * (~pepper), im)

mh.imsave('../1400OS_10_13+.jpg', im.astype(np.uint8))
