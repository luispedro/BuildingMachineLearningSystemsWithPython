# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

import numpy as np
import mahotas as mh
image = mh.imread('../SimpleImageDataset/building05.jpg')
image = mh.colors.rgb2gray(image, dtype=np.uint8)

# We need to downsample ``image`` so that the details are visibly pixelated.
# This exaggerates the effect so that the result is clear
image = image[::4, ::4]
thresh = mh.sobel(image)
filtered = mh.sobel(image, just_filter=True)

thresh = mh.dilate(thresh, np.ones((7, 7)))
filtered = mh.dilate(mh.stretch(filtered), np.ones((7, 7)))

# Paste the thresholded and non-thresholded versions of the image side-by-side
h, w = thresh.shape
canvas = 255 * np.ones((h, w * 2 + 64), np.uint8)
canvas[:, :w] = thresh * 255
canvas[:, -w:] = filtered

mh.imsave('../1400OS_10_09+.jpg', canvas)
