# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

import numpy as np
import mahotas as mh

text = mh.imread("simple-dataset/text21.jpg")
scene = mh.imread("simple-dataset/scene00.jpg")
h, w, _ = text.shape
canvas = np.zeros((h, 2 * w + 128, 3), np.uint8)
canvas[:, -w:] = scene
canvas[:, :w] = text
canvas = canvas[::4, ::4]
mh.imsave('../1400OS_10_10+.jpg', canvas)
