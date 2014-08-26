=========
Chapter 4
=========

Support code for *Chapter 4: Topic Modeling*

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
feasible to run it on a modern laptop.

