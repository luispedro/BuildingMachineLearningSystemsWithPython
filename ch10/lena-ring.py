# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

import mahotas as mh
import numpy as np

# Read in the image
im = mh.demos.load('lena')

# This breaks up the image into RGB channels
r, g, b = im.transpose(2, 0, 1)
h, w = r.shape

# smooth the image per channel:
r12 = mh.gaussian_filter(r, 12.)
g12 = mh.gaussian_filter(g, 12.)
b12 = mh.gaussian_filter(b, 12.)

# build back the RGB image
im12 = mh.as_rgb(r12, g12, b12)

X, Y = np.mgrid[:h, :w]
X = X - h / 2.
Y = Y - w / 2.
X /= X.max()
Y /= Y.max()

# Array C will have the highest values in the center, fading out to the edges:

C = np.exp(-2. * (X ** 2 + Y ** 2))
C -= C.min()
C /= C.ptp()
C = C[:, :, None]

# The final result is sharp in the centre and smooths out to the borders:
ring = mh.stretch(im * C + (1 - C) * im12)
mh.imsave('lena-ring.jpg', ring)
