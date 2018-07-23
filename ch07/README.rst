=========
Chapter 8
=========

Support code for *Chapter 8: Recommendations*.

The code refers to the second edition of the book and this code has been
significantly refactored when compared to the first one.

Ratings Prediction
------------------

Note that since the partition of the data into training and testing is random,
everytime you run the code, the results will be different.


load_ml100k.py
    Load data & partition into test/train
norm.py
    Normalize the data
corrneighbours.py
    Neighbour models based on ncrroaltoin
regression.py
    Regression models
stacked.py
    Stacked predictions
averaged.py
    Averaging of predictions (mentioned in book, but code is not shown there).

Association Rule Mining
-----------------------

Check the folder ``apriori/``

apriori/histogram.py
    Print a histogram of how many times each product was bought
apriori/apriori.py
    Implementation of Apriori algorithm and association rule building
apriori/apriori_example.py
    Example of Apriori algorithm in retail dataset

