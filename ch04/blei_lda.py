# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

from __future__ import print_function
try:
    from gensim import corpora, models
except:
    print("import gensim failed.")
    print()
    print("Please install it")
    raise

try:
    from mpltools import style
    style.use('ggplot')
except:
    print("Could not import mpltools: plots will not be styled correctly")

import matplotlib.pyplot as plt
import numpy as np
from os import path

NUM_TOPICS = 100

# Check that data exists
if not path.exists('./data/ap/ap.dat'):
    print('Error: Expected data to be present at data/ap/')
    print('Please cd into ./data & run ./download_ap.sh')

# Load the data
corpus = corpora.BleiCorpus('./data/ap/ap.dat', './data/ap/vocab.txt')

# Build the topic model
model = models.ldamodel.LdaModel(
    corpus, num_topics=NUM_TOPICS, id2word=corpus.id2word, alpha=None)

ti = 0
# Iterate over all the topics in the model
for ti in xrange(model.num_topics):
    words = model.show_topic(ti, 64)
    tf = sum(f for f, w in words)
    print('\n'.join('{}:{}'.format(w, int(1000. * f / tf)) for f, w in words))
    print()
    print()
    print()

try:
    from pytagcloud import create_tag_image, make_tags

    # We first identify the most discussed topic, i.e., the one with the
    # highest total weight

    # First, we need to sum up the weights across all the documents
    weight = np.zeros(model.num_topics)
    for doc in corpus:
        for col, val in model[doc]:
            weight[col] += val
            # As a reasonable alternative, we could have used the log of val:
            # weight[col] += np.log(val)
    max_topic = weight.argmax()

    # Get the top 64 words for this topic
    # Without the argument, show_topic would return only 10 words
    words = model.show_topic(max_topic, 64)

    # gensim returns a weight between 0 and 1 for each word, while pytagcloud
    # expects an integer word count. So, we multiply by a large number and
    # round. For a visualization this is an adequate approximation.
    # We also need to flip the order as gensim returns (value, word), whilst
    # pytagcloud expects (word, value):
    words = [(w,int(v*10000)) for v,w in words]
    tags = make_tags(words, maxsize=120)
    create_tag_image(tags, 'cloud_large.png', size=(1800, 1200), fontname='Lobster')
except ImportError:
    print("Could not import pytagcloud. Skipping clooud generation")

num_topics_used = [len(model[doc]) for doc in corpus]
plt.hist(num_topics_used, np.arange(42))
plt.ylabel('Nr of documents')
plt.xlabel('Nr of topics')
plt.savefig('../1400OS_04_01+.png')
plt.clf()


# Now, repeat the same exercise using alpha=1:

model1 = models.ldamodel.LdaModel(
    corpus, num_topics=NUM_TOPICS, id2word=corpus.id2word, alpha=1.)
num_topics_used1 = [len(model1[doc]) for doc in corpus]

plt.hist([num_topics_used, num_topics_used1], np.arange(42))
plt.ylabel('Nr of documents')
plt.xlabel('Nr of topics')

# The coordinates below were fit by trial and error to look good
plt.text(9, 223, r'default alpha')
plt.text(26, 156, 'alpha=1.0')
plt.savefig('../1400OS_04_02+.png')
