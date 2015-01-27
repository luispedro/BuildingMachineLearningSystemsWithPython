# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License
from __future__ import print_function

try:
    import nltk.corpus
except ImportError:
    print("nltk not found")
    print("please install it")
    raise
from scipy.spatial import distance
import numpy as np
from gensim import corpora, models
import sklearn.datasets
import nltk.stem
from collections import defaultdict

english_stemmer = nltk.stem.SnowballStemmer('english')
stopwords = set(nltk.corpus.stopwords.words('english'))
stopwords.update(['from:', 'subject:', 'writes:', 'writes'])


class DirectText(corpora.textcorpus.TextCorpus):

    def get_texts(self):
        return self.input

    def __len__(self):
        return len(self.input)
try:
    dataset = sklearn.datasets.load_mlcomp("20news-18828", "train",
                                       mlcomp_root='./data')
except:
    print("Newsgroup data not found.")
    print("Please download from http://mlcomp.org/datasets/379")
    print("And expand the zip into the subdirectory data/")
    print()
    print()
    raise

otexts = dataset.data
texts = dataset.data

texts = [t.decode('utf-8', 'ignore') for t in texts]
texts = [t.split() for t in texts]
texts = [map(lambda w: w.lower(), t) for t in texts]
texts = [filter(lambda s: not len(set("+-.?!()>@012345689") & set(s)), t)
         for t in texts]
texts = [filter(lambda s: (len(s) > 3) and (s not in stopwords), t)
         for t in texts]
texts = [map(english_stemmer.stem, t) for t in texts]
usage = defaultdict(int)
for t in texts:
    for w in set(t):
        usage[w] += 1
limit = len(texts) / 10
too_common = [w for w in usage if usage[w] > limit]
too_common = set(too_common)
texts = [filter(lambda s: s not in too_common, t) for t in texts]

corpus = DirectText(texts)
dictionary = corpus.dictionary
try:
    dictionary['computer']
except:
    pass

model = models.ldamodel.LdaModel(
    corpus, num_topics=100, id2word=dictionary.id2token)

thetas = np.zeros((len(texts), 100))
for i, c in enumerate(corpus):
    for ti, v in model[c]:
        thetas[i, ti] += v

distances = distance.squareform(distance.pdist(thetas))
large = distances.max() + 1
for i in range(len(distances)):
    distances[i, i] = large

print(otexts[1])
print()
print()
print()
print(otexts[distances[1].argmin()])
