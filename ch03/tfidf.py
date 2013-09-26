# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 15:50:11 2012

@author: wilrich
"""

import scipy as sp


def tfidf(t, d, D):
    tf = float(d.count(t)) / sum(d.count(w) for w in set(d))
    idf = sp.log(float(len(D)) / (len([doc for doc in D if t in doc])))
    return tf * idf


a, abb, abc = ["a"], ["a", "b", "b"], ["a", "b", "c"]
D = [a, abb, abc]

print(tfidf("a", a, D))
print(tfidf("b", abb, D))
print(tfidf("a", abc, D))
print(tfidf("b", abc, D))
print(tfidf("c", abc, D))
