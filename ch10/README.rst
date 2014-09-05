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

threshold.py
    Threshold building image with both Otsu and Ridley-Calvard threshold
lena-ring.py
    Show Lena in a sharp ring around a soft focus image
figure5_6.py
    Computes figures 5 and 6 in the book
figure9.py
    Compute Sobel images
figure10.py
    Just paste two images next to each others
figure13.py
    Demonstrate salt&pepper effect
features.py
    Contains the ``edginess_sobel`` function from the book as well as a simple
    wrapper around ``mahotas.texture.haralick``
simple_classification.py
    Classify SimpleImageDataset with texture features + sobel feature
figure18.py
    Classify ``AnimTransDistr`` with both texture and SURF features.

