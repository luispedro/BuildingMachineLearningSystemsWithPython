# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

from __future__ import print_function
import numpy as np
import logging
import gensim
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO)
id2word = gensim.corpora.Dictionary.load_from_text(
    'data/wiki_en_output_wordids.txt')
mm = gensim.corpora.MmCorpus('data/wiki_en_output_tfidf.mm')
model = gensim.models.ldamodel.LdaModel(
    corpus=mm,
    id2word=id2word,
    num_topics=100,
    update_every=1,
    chunksize=10000,
    passes=1)
model.save('wiki_lda.pkl')
topics = [model[doc] for doc in mm]
lens = np.array([len(t) for t in topics])
print(np.mean(lens <= 10))
print(np.mean(lens))

counts = np.zeros(100)
for doc_top in topics:
    for ti, _ in doc_toc:
        counts[ti] += 1

for doc_top in topics:
    for ti, _ in doc_top:
        counts[ti] += 1

words = model.show_topic(counts.argmax(), 64)
print(words)
print()
print()
print()
words = model.show_topic(counts.argmin(), 64)
print(words)
print()
print()
print()
