# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

from __future__ import print_function
import numpy as np
import gensim
from os import path
from wordcloud import create_cloud

if not path.exists('wiki_lda.pkl'):
    import sys
    sys.stderr.write('''\
This script must be run after wikitopics_create.py!

That script creates and saves the LDA model (this must onlly be done once).
This script is responsible for the analysis.''')
    sys.exit(1)

# Load the preprocessed Wikipedia corpus (id2word and mm)
id2word = gensim.corpora.Dictionary.load_from_text(
    'data/wiki_en_output_wordids.txt.bz2')
mm = gensim.corpora.MmCorpus('data/wiki_en_output_tfidf.mm')

# Load the precomputed model
model = gensim.models.ldamodel.LdaModel.load('wiki_lda.pkl')

# Compute topics for all elements of the model:
topics = [model[doc] for doc in mm]
lens = np.array([len(t) for t in topics])
print('Mean number of topics mentioned: {0:.3}'.format(np.mean(lens)))
print('Percentage of articles mentioning less than 10 topics: {0:.1%}'.format(np.mean(lens <= 10)))


counts = np.zeros(100)
for doc_top in topics:
    for ti,tv in doc_top:
        counts[ti] += tv

# Retrieve the most heavily used topic and plot it as a word cloud:
words = model.show_topic(counts.argmax(), 64)
create_cloud('Wikipedia_most.png', words)
print(words)
print()
print()
print()

# Retrieve the **least** heavily used topic and plot it as a word cloud:
words = model.show_topic(counts.argmin(), 64)
create_cloud('Wikipedia_least.png', words)
print(words)
print()
print()
print()
