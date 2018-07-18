==========
Chapter 10
==========

Support code for *Chapter 10: Pattern Recognition & Computer Vision*

Data
----

This chapter relies on a publicly available dataset (which can be downloaded
using the ``download.sh`` script inside the ``data/`` directory) as well the
dataset that is packaged with the repository at ``../SimpleImageDataset/``.

Running ``download.sh`` will retrieve the other dataset into a directory
``AnimTransDistr/``.

Scripts
-------

chapter.py
    Code as written in the book.
thresholded_figure.py
    Computes the thresholded figures, including after Gaussian blurring
lena-ring.py
    Lena image with center in focus and blurred edges
figure10.py
    Just paste two images next to each others
features.py
    Contains the color histogram function from the book as well as a simple
    wrapper around ``mahotas.texture.haralick``
simple_classification.py
    Classify SimpleImageDataset with texture features + color histogram features
large_classification.py
    Classify ``AnimTransDistr`` with both texture and SURF features.
neighbors.py
    Computes image neighbors as well as the neighbor figure from the book.

