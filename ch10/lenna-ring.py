# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

import mahotas as mh
import numpy as np
im = mh.imread('lenna.jpg')
r, g, b = im.transpose(2, 0, 1)
h, w = r.shape
r12 = mh.gaussian_filter(r, 12.)
g12 = mh.gaussian_filter(g, 12.)
b12 = mh.gaussian_filter(b, 12.)
im12 = mh.as_rgb(r12, g12, b12)

X, Y = np.mgrid[:h, :w]
X = X - h / 2.
Y = Y - w / 2.
X /= X.max()
Y /= Y.max()
C = np.exp(-2. * (X ** 2 + Y ** 2))
C -= C.min()
C /= C.ptp()
C = C[:, :, None]

ring = mh.stretch(im * C + (1 - C) * im12)
mh.imsave('lenna-ring.jpg', ring)
