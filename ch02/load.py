# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

import numpy as np


def load_dataset(dataset_name):
    '''
    data = load_dataset(dataset_name)

    Load a given dataset

    Returns
    -------
    data : dictionary
    '''
    features = []
    target = []
    target_names = set()
    with open('./data/{0}.tsv'.format(dataset_name)) as ifile:
        for line in ifile:
            tokens = line.strip().split('\t')
            features.append([float(tk) for tk in tokens[:-1]])
            target.append(tokens[-1])
            target_names.add(tokens[-1])
    features = np.array(features)

    target_names = list(target_names)
    target_names.sort()
    target = np.array([target_names.index(t) for t in target])
    return {
            'features': features,
            'target_names': target_names,
            'target': target,
            }
