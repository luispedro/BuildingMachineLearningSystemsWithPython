=========
Chapter 2
=========

Support code for *Chapter 2: Learning How to Classify with Real-world
Examples*. The directory data contains the seeds dataset, originally downloaded
from https://archive.ics.uci.edu/ml/datasets/seeds

chapter.py
    The code as printed in the book.

figure1.py
    Figure 1 in the book: all 2-by-2 scatter plots

figure2.py
    Figure 2 in the book: threshold & decision area

figure4_5_sklearn.py
    Figures 4 and 5 in the book: Knn decision borders before and after feature
    normalization. This also produces a version of the figure using 11
    neighbors (not in the book), which shows that the result is smoother, not
    as sensitive to exact positions of each datapoint.

figure4_5_no_sklearn.py
    Alternative code for Figures 4 and 5 without using scikit-learn
    
load.py
    Code to load the seeds data

simple_threshold.py
    Code from the book: finds the first partition, between Setosa and the other classes.

stump.py
    Code from the book: finds the second partition, between Virginica and Versicolor.

threshold.py
    Functional implementation of a threshold classifier

heldout.py
    Evalute the threshold model on heldout data

seeds_knn_sklearn.py
    Demonstrate cross-validation and feature normalization using scikit-learn
    
seeds_threshold.py
    Test thresholding model on the seeds dataset (result mention in book, but no code)

seeds_knn_increasing_k.py
    Test effect of increasing num_neighbors on accuracy.

knn.py
    Implementation of K-Nearest neighbor without using scikit-learn.

seeds_knn.py
    Demonstrate cross-validation (without scikit-learn)
