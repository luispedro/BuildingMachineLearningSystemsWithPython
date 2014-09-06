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

topics = np.load('topics.npy', mmap_mode='r')

# Compute the number of topics mentioned in each document
lens = (topics > 0).sum(axis=1)
print('Mean number of topics mentioned: {0:.3}'.format(np.mean(lens)))
print('Percentage of articles mentioning less than 10 topics: {0:.1%}'.format(np.mean(lens <= 10)))

# Weights will be the total weight of each topic
weights = topics.sum(0)

# Retrieve the most heavily used topic and plot it as a word cloud:
words = model.show_topic(weights.argmax(), 64)

# The parameter ``maxsize`` often needs some manual tuning to make it look nice.
create_cloud('Wikipedia_most.png', words, maxsize=250, fontname='Cardo')

fraction_mention = np.mean(topics[:,weights.argmax()] > 0)
print("The most mentioned topics is mentioned in {:.1%} of documents.".format(fraction_mention))
total_weight = np.mean(topics[:,weights.argmax()])
print("It represents {:.1%} of the total number of words.".format(total_weight))
print()
print()
print()

# Retrieve the **least** heavily used topic and plot it as a word cloud:
words = model.show_topic(weights.argmin(), 64)
create_cloud('Wikipedia_least.png', words, maxsize=150, fontname='Cardo')
fraction_mention = np.mean(topics[:,weights.argmin()] > 0)
print("The least mentioned topics is mentioned in {:.1%} of documents.".format(fraction_mention))
total_weight = np.mean(topics[:,weights.argmin()])
print("It represents {:.1%} of the total number of words.".format(total_weight))
print()
print()
print()
