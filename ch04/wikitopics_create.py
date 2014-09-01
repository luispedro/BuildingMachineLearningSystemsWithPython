# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

from __future__ import print_function
import logging
import gensim
import numpy as np

NR_OF_TOPICS = 100

# Set up logging in order to get progress information as the model is being built:
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO)

# Load the preprocessed corpus (id2word & mm):
id2word = gensim.corpora.Dictionary.load_from_text(
    'data/wiki_en_output_wordids.txt.bz2')
mm = gensim.corpora.MmCorpus('data/wiki_en_output_tfidf.mm')

# Calling the constructor is enough to build the model
# This call will take a few hours!
model = gensim.models.ldamodel.LdaModel(
    corpus=mm,
    id2word=id2word,
    num_topics=NR_OF_TOPICS,
    update_every=1,
    chunksize=10000,
    passes=1)

# Save the model so we do not need to learn it again.
model.save('wiki_lda.pkl')

# Compute the document/topic matrix
topics = np.zeros((len(mm), model.num_topics))
for di,doc in enumerate(mm):
    doc_top = model[doc]
    for ti,tv in doc_top:
        topics[di,ti] += tv
np.save('topics.npy', topics)

# Alternatively, we create a sparse matrix and save that. This alternative
# saves disk space, at the cost of slightly more complex code:

## from scipy import sparse, io
## sp = sparse.csr_matrix(topics)
## io.savemat('topics.mat', {'topics': sp})
