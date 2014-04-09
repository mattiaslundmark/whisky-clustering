import numpy as np
import pandas as pd
import pylab as pl

from sklearn.cluster import KMeans
from sklearn import preprocessing as prep
from sklearn.decomposition import PCA

from optparse import OptionParser

op = OptionParser()
op.add_option("--nbr_clusters", dest="nbr_clusters", type="int", default="6")

(opts, args) = op.parse_args()

nbr_clusters = opts.nbr_clusters

pd_whiskies = pd.read_csv('whiskies.txt')
whiskies_full = pd_whiskies.as_matrix()
whiskies = pd_whiskies.as_matrix()
coordinates = whiskies[:,15:17].astype(float)
coordinates = prep.scale(coordinates)

fig = pl.figure()
pl.plot(coordinates[:,0], coordinates[:,1], 'o')

# subset with taste data
whiskies = whiskies[:,2:13]
whiskies = whiskies.astype(float)
print(whiskies)

# reduce dimensions for plotting
pca = PCA(n_components=2)
pca.fit(whiskies)
whiskies_pca = pca.transform(whiskies)
print(whiskies_pca)

# scale
whiskies = prep.scale(whiskies)
print(whiskies)

cost_array = np.zeros(20)

for i in xrange(1,20):
    km = KMeans(i, init='k-means++', max_iter=100, n_init=1, verbose=True)
    cost = km.fit(whiskies)
    cost_array[i] = cost.inertia_
    print(cost.inertia_)
    
print(cost_array)

fig = pl.figure()
pl.plot(cost_array[1:20])

km = KMeans(nbr_clusters, init='k-means++', max_iter=100, n_init=1, verbose=True)
cost = km.fit(whiskies)
print(cost.inertia_)
print(cost.cluster_centers_)

clusters = np.zeros(nbr_clusters)
fig = pl.figure()
for c in xrange(nbr_clusters):
    cluster_members = cost.labels_ == c
    pl.plot(whiskies_pca[cluster_members, 0], whiskies_pca[cluster_members, 1], 'o')
    print("Members of cluster %d" % c)
    #print(cluster_members)
    print(whiskies_full[cluster_members, 1])

pl.show()

