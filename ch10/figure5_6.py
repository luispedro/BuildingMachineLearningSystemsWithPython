# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

from matplotlib import pyplot as plt
import numpy as np
import mahotas as mh
image = mh.imread('../1400OS_10_01.jpeg')
image = mh.colors.rgb2gray(image)
im8 = mh.gaussian_filter(image, 8)
im16 = mh.gaussian_filter(image, 16)
im32 = mh.gaussian_filter(image, 32)
h, w = im8.shape
canvas = np.ones((h, 3 * w + 256), np.uint8)
canvas *= 255
canvas[:, :w] = im8
canvas[:, w + 128:2 * w + 128] = im16
canvas[:, -w:] = im32
mh.imsave('../1400OS_10_05+.jpg', canvas[:, ::2])

im32 = mh.stretch(im32)
ot32 = mh.otsu(im32)
mh.imsave('../1400OS_10_06+.jpg', (im32 > ot32).astype(np.uint8) * 255)
