=========
Chapter 4
=========

Support code for *Chapter 4: Topic Modeling*


AP Data
-------

To download the AP data, use the ``download_ap.sh`` script inside the ``data``
directory::

    cd data
    ./download_ap.sh

Word cloud creation
-------------------

Word cloud creation requires that ``pytagcloud`` be installed (in turn, this
requires ``pygame``). Since this is not an essential part of the chapter, the
code will work even if you have not installed it (naturally, the cloud image
will not be generated and a warning will be printed).


Wikipedia processing
--------------------

You will need **a lot of disk space**. The download of the Wikipedia text is
11GB and preprocessing it takes another 24GB to save it in the intermediate
format that gensim uses for a total of 34GB!

Run the following two commands inside the ``data/`` directory::

    ./download_wp.sh
    ./preprocess-wikidata.sh

As the filenames indicate, the first step will download the data and the second
one will preprocess it. Preprocessing can take several hours, but it is
feasible to run it on a modern laptop. Once the second step is finished, you
may remove the input file if you want to save disk space
(``data/enwiki-latest-pages-articles.xml.bz2``).

To generate the model, you can run the ``wikitopics_create.py`` script, while
the ``wikitopics_plot.py`` script will plot the most heavily discussed topic as
well as the least heavily discussed one. The code is split into steps as the
first one can take a very long time. Then it saves the results so that you can
later explore them at leisure.

You should not expect that your results will exactly match the results in the
book, for two reasons:

1. The LDA algorithm is a probabilistic algorithm and can give different
   results every time it is run.
2. Wikipedia keeps changing. Thus, even your input data will be different.

Scripts
-------

blei_lda.py
    Computes LDA using the AP Corpus.
wikitopics_create.py
    Create the topic model for Wikipedia using LDA (must download wikipedia database first)
wikitopics_create_hdp.py
    Create the topic model for Wikipedia using HDP (must download wikipedia database first)
