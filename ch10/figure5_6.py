# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

import numpy as np
import mahotas as mh
image = mh.imread('../SimpleImageDataset/building05.jpg')
image = mh.colors.rgb2gray(image)

# Compute Gaussian filtered versions with increasing kernel widths
im8  = mh.gaussian_filter(image,  8)
im16 = mh.gaussian_filter(image, 16)
im32 = mh.gaussian_filter(image, 32)

# We now build a composite image with three panels:
#
# [ IM8 | | IM16 | | IM32 ]

h, w = im8.shape
canvas = np.ones((h, 3 * w + 256), np.uint8)
canvas *= 255
canvas[:, :w] = im8
canvas[:, w + 128:2 * w + 128] = im16
canvas[:, -w:] = im32
mh.imsave('../1400OS_10_05+.jpg', canvas[:, ::2])

# Threshold the image
# We need to first stretch it to convert to an integer image
im32 = mh.stretch(im32)
ot32 = mh.otsu(im32)

# Convert to 255 np.uint8 to match the other images
im255 = 255 * (im32 > ot32).astype(np.uint8)
mh.imsave('../1400OS_10_06+.jpg', im255)
