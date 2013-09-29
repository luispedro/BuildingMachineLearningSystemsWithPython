# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

from load_ml100k import load
from matplotlib import pyplot as plt
data = load()
data = data.toarray()
plt.gray()
plt.imshow(data[:200, :200], interpolation='nearest')
plt.xlabel('User ID')
plt.ylabel('Film ID')
plt.savefig('../1400_08_03+.png')
