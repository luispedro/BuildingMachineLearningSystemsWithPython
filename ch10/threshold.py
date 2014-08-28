# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

import numpy as np
import mahotas as mh

# Load our example image:
image = mh.imread('../SimpleImageDataset/building05.jpg')

# Convert to greyscale
image = mh.colors.rgb2gray(image, dtype=np.uint8)

# Compute a threshold value:
thresh = mh.thresholding.otsu(image)
print('Otsu threshold is {0}'.format(thresh))

# Compute the thresholded image
otsubin = (image > thresh)
print('Saving thresholded image (with Otsu threshold) to otsu-threshold.jpeg')
mh.imsave('otsu-threshold.jpeg', otsubin.astype(np.uint8) * 255)

# Execute morphological opening to smooth out the edges
otsubin = mh.open(otsubin, np.ones((15, 15)))
mh.imsave('otsu-closed.jpeg', otsubin.astype(np.uint8) * 255)

# An alternative thresholding method:
thresh = mh.thresholding.rc(image)
print('Ridley-Calvard threshold is {0}'.format(thresh))
print('Saving thresholded image (with Ridley-Calvard threshold) to rc-threshold.jpeg')
mh.imsave('rc-threshold.jpeg', (image > thresh).astype(np.uint8) * 255)
