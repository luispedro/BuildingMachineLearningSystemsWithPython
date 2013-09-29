# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

from __future__ import print_function
from gensim import corpora, models, similarities
from mpltools import style
import matplotlib.pyplot as plt
import numpy as np
from os import path
style.use('ggplot')

if not path.exists('./data/ap/ap.dat'):
    print('Error: Expected data to be present at data/ap/')

corpus = corpora.BleiCorpus('./data/ap/ap.dat', './data/ap/vocab.txt')
model = models.ldamodel.LdaModel(
    corpus, num_topics=100, id2word=corpus.id2word, alpha=None)

for ti in xrange(84):
    words = model.show_topic(ti, 64)
    tf = sum(f for f, w in words)
    print('\n'.join('{}:{}'.format(w, int(1000. * f / tf)) for f, w in words))
    print()
    print()
    print()

thetas = [model[c] for c in corpus]
plt.hist([len(t) for t in thetas], np.arange(42))
plt.ylabel('Nr of documents')
plt.xlabel('Nr of topics')
plt.savefig('../1400OS_04_01+.png')

model1 = models.ldamodel.LdaModel(
    corpus, num_topics=100, id2word=corpus.id2word, alpha=1.)
thetas1 = [model1[c] for c in corpus]

#model8 = models.ldamodel.LdaModel(corpus, num_topics=100, id2word=corpus.id2word, alpha=1.e-8)
#thetas8 = [model8[c] for c in corpus]
plt.clf()
plt.hist([[len(t) for t in thetas], [len(t) for t in thetas1]], np.arange(42))
plt.ylabel('Nr of documents')
plt.xlabel('Nr of topics')
plt.text(9, 223, r'default alpha')
plt.text(26, 156, 'alpha=1.0')
plt.savefig('../1400OS_04_02+.png')
