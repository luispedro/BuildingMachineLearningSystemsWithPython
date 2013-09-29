# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

import numpy as np
from collections import defaultdict
from itertools import chain
from gzip import GzipFile
dataset = [[int(tok) for tok in line.strip().split()]
           for line in GzipFile('retail.dat.gz')]
counts = defaultdict(int)
for elem in chain(*dataset):
    counts[elem] += 1
counts = np.array(list(counts.values()))
bins = [1, 2, 4, 8, 16, 32, 64, 128, 512]
print(' {0:11} | {1:12}'.format('Nr of baskets', 'Nr of products'))
print('--------------------------------')
for i in range(len(bins)):
    bot = bins[i]
    top = (bins[i + 1] if (i + 1) < len(bins) else 100000000000)
    print('  {0:4} - {1:3}   | {2:12}'.format(
        bot, (top if top < 1000 else ''), np.sum((counts >= bot) & (counts < top))))
