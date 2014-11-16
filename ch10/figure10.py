# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

import numpy as np
import mahotas as mh

# This little script just builds an image with two examples, side-by-side:

text = mh.imread("../SimpleImageDataset/text21.jpg")
building = mh.imread("../SimpleImageDataset/building00.jpg")
h, w, _ = text.shape
canvas = np.zeros((h, 2 * w + 128, 3), np.uint8)
canvas[:, -w:] = building
canvas[:, :w] = text
canvas = canvas[::4, ::4]
mh.imsave('figure10.jpg', canvas)
