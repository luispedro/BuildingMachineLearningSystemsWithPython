# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

import numpy as np
import mahotas as mh
image = mh.imread('../1400OS_10_01.jpeg')
image = mh.colors.rgb2gray(image, dtype=np.uint8)
thresh = mh.thresholding.otsu(image)
print(thresh)
otsubin = (image > thresh)
mh.imsave('otsu-threshold.jpeg', otsubin.astype(np.uint8) * 255)
otsubin = ~ mh.close(~otsubin, np.ones((15, 15)))
mh.imsave('otsu-closed.jpeg', otsubin.astype(np.uint8) * 255)

thresh = mh.thresholding.rc(image)
print(thresh)
mh.imsave('rc-threshold.jpeg', (image > thresh).astype(np.uint8) * 255)
