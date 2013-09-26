import numpy as np
import mahotas as mh
def edginess_sobel(image):
    '''
    edgi = edginess_sobel(image)

    Measure the "edginess" of an image
    '''
    edges = mh.sobel(image, just_filter=True)
    edges = edges.ravel()
    return np.sqrt(np.dot(edges, edges))
