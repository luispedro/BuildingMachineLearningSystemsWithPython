COLOUR_FIGURE = False

from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from load import load_dataset
import numpy as np
from knn import learn_model, apply_model, accuracy

feature_names = [
    'area',
    'perimeter',
    'compactness',
    'length of kernel',
    'width of kernel',
    'asymmetry coefficien',
    'length of kernel groove',
]


def train_plot(features, labels):
    y0,y1 = features[:,2].min()*.9, features[:,2].max()*1.1
    x0,x1 = features[:,0].min()*.9, features[:,0].max()*1.1
    X = np.linspace(x0,x1,100)
    Y = np.linspace(y0,y1,100)
    X,Y = np.meshgrid(X,Y)

    model = learn_model(1, features[:,(0,2)], np.array(labels))
    C = apply_model(np.vstack([X.ravel(),Y.ravel()]).T, model).reshape(X.shape)
    if COLOUR_FIGURE:
        cmap = ListedColormap([(1.,.6,.6),(.6,1.,.6),(.6,.6,1.)])
    else:
        cmap = ListedColormap([(1.,1.,1.),(.2,.2,.2),(.6,.6,.6)])
    plt.xlim(x0,x1)
    plt.ylim(y0,y1)
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[2])
    plt.pcolormesh(X,Y,C, cmap=cmap)
    if COLOUR_FIGURE:
        cmap = ListedColormap([(1.,.0,.0),(.0,1.,.0),(.0,.0,1.)])
        plt.scatter(features[:,0], features[:,2], c=labels, cmap=cmap)
    else:
        for lab,ma in zip(range(3), "Do^"):
            plt.plot(features[labels == lab,0], features[labels == lab,2], ma, c=(1.,1.,1.))


features,labels = load_dataset('seeds')
names = sorted(set(labels))
labels = np.array([names.index(ell) for ell in labels])

train_plot(features, labels)
plt.savefig('../1400_02_04.png')

features -= features.mean(0)
features /= features.std(0)
train_plot(features, labels)
plt.savefig('../1400_02_05.png')
