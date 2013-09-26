COLOUR_FIGURE = False

from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
data = load_iris()
features = data['data']
feature_names = data['feature_names']
species = data['target_names'][data['target']]

setosa = (species == 'setosa')
features = features[~setosa]
species = species[~setosa]
virginica = species == 'virginica'

t = 1.75
p0,p1 = 3,2

if COLOUR_FIGURE:
    area1c = (1.,.8,.8)
    area2c = (.8,.8,1.)
else:
    area1c = (1.,1,1)
    area2c = (.7,.7,.7)

x0,x1 =[features[:,p0].min()*.9,features[:,p0].max()*1.1]
y0,y1 =[features[:,p1].min()*.9,features[:,p1].max()*1.1]

plt.fill_between([t,x1],[y0,y0],[y1,y1],color=area2c)
plt.fill_between([x0,t],[y0,y0],[y1,y1],color=area1c)
plt.plot([t,t],[y0,y1],'k--',lw=2)
plt.plot([t-.1,t-.1],[y0,y1],'k:',lw=2)
plt.scatter(features[virginica,p0], features[virginica,p1], c='b', marker='o')
plt.scatter(features[~virginica,p0], features[~virginica,p1], c='r', marker='x')
plt.ylim(y0,y1)
plt.xlim(x0,x1)
plt.xlabel(feature_names[p0])
plt.ylabel(feature_names[p1])
plt.savefig('../1400_02_02.png')
