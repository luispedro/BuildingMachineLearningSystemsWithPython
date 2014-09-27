=========
Chapter 7
=========

Support code for *Chapter 7: Regression* 


Boston data analysis
--------------------

This dataset is shipped with sklearn. Thus, no extra download is required.


boston1.py
    Fit a linear regression model to the Boston house price data
boston1numpy.py
    Version of above script using numpy operations for linear regression
boston_cv_penalized.py
    Test different penalized (and OLS) regression schemes on the Boston dataset
figure1_2.py
    Show the regression line for Boston data
figure3.py
    Show the regression line for Boston data with OLS and Lasso
figure4.py
    Scatter plot of predicted-vs-actual for multidimensional regression

10K data analysis
-----------------

lr10k.py
    Linear regression on 10K dataset, evaluation by cross-validation
predict10k_en.py
    Elastic nets (including with inner cross-validation for parameter
    settings). Produces scatter plot.


MovieLens data analysis
-----------------------

In this chapter, we only consider a very simple approach, which is implemented
in the ``usermodel.py`` script.

