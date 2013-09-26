import mahotas as mh
from mahotas.colors import rgb2grey
import numpy as np

im = mh.imread('lenna.jpg')
im = rgb2grey(im)

salt = np.random.random(im.shape) > .975
pepper = np.random.random(im.shape) > .975

im = np.maximum(salt*170, mh.stretch(im))
im = np.minimum(pepper*30 + im*(~pepper), im)

mh.imsave('../1400OS_10_13+.jpg', im.astype(np.uint8))
