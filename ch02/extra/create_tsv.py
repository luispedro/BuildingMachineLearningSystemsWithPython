# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

import milksets.seeds


def save_as_tsv(fname, module):
    features, labels = module.load()
    nlabels = [module.label_names[ell] for ell in labels]
    with open(fname, 'w') as ofile:
        for f, n in zip(features, nlabels):
            print >>ofile, "\t".join(map(str, f) + [n])

save_as_tsv('seeds.tsv', milksets.seeds)
