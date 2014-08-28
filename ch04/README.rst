=========
Chapter 4
=========

Support code for *Chapter 4: Topic Modeling*

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

