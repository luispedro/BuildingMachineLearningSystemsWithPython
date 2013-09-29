# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

from apriori import apriori, association_rules
from gzip import GzipFile
dataset = [[int(tok) for tok in line.strip().split()]
           for line in GzipFile('retail.dat.gz')]
freqsets, baskets = apriori(dataset, 80, maxsize=5)
nr_transactions = float(len(dataset))
for ant, con, base, pyx, lift in association_rules(dataset, freqsets, baskets, 30):
    print('{} | {} | {} ({:%}) | {} | {} | {}'
          .format(ant, con, len(baskets[con]), len(baskets[con]) / nr_transactions, len(baskets[ant]), len(baskets[con | ant]), int(lift)))
