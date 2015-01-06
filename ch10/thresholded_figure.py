import mahotas as mh
import numpy as np
from matplotlib import pyplot as plt

# Load image & convert to B&W
image = mh.imread('../SimpleImageDataset/scene00.jpg')
image = mh.colors.rgb2grey(image, dtype=np.uint8)
plt.imshow(image)
plt.gray()
plt.title('original image')

thresh = mh.thresholding.otsu(image)
print('Otsu threshold is {}.'.format(thresh))

threshed = (image > thresh)
plt.figure()
plt.imshow(threshed)
plt.title('threholded image')
mh.imsave('thresholded.png', threshed.astype(np.uint8)*255)

im16 = mh.gaussian_filter(image, 16)

# Repeat the thresholding operations with the blurred image 
thresh = mh.thresholding.otsu(im16.astype(np.uint8))
threshed = (im16 > thresh)
plt.figure()
plt.imshow(threshed)
plt.title('threholded image (after blurring)')
print('Otsu threshold after blurring is {}.'.format(thresh))
mh.imsave('thresholded16.png', threshed.astype(np.uint8)*255)
plt.show()
