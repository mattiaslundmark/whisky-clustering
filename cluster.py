import numpy as np
import pandas as pd
import pylab as pl
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import preprocessing as prep
from sklearn.decomposition import PCA

from optparse import OptionParser

op = OptionParser()
op.add_option("--nbr_clusters", dest="nbr_clusters", type="int", default="6")

(opts, args) = op.parse_args()

nbr_clusters = opts.nbr_clusters

# Read data from csv and store it in a matrix
pd_whiskies = pd.read_csv('whiskies.txt')
whiskies_full = pd_whiskies.as_matrix()
whiskies = pd_whiskies.as_matrix()

# Plot coordinates
# TODO overlay on a map of Scotland
coordinates = whiskies[:,15:17].astype(float)
coordinates = prep.scale(coordinates)
#fig = pl.figure()
#pl.plot(coordinates[:,0], coordinates[:,1], 'o')

# subset with taste data as floats. All rows and columns 2 to 13. Name and coordinates of no use here
whiskies = whiskies[:,2:13]
whiskies = whiskies.astype(float)
print(whiskies)

# Reduce dimensions for plotting since plotting 12 dimensions is relatively hard.
pca = PCA(n_components=2)
whiskies_pca = pca.fit_transform(whiskies)
print(whiskies_pca)

# scaling, not really necessary here
# whiskies = prep.scale(whiskies)
# print(whiskies)

# Initialize an array to keep track of the cost (the result) of a fit with a number of clusters
cost_array = np.zeros(20)

# Run the KMeans algorith for 1 up to 20 clusters
for i in range(1,20):
    km = KMeans(i, init='k-means++', max_iter=100, n_init=1, verbose=True)
    result = km.fit(whiskies)
    cost_array[i] = result.inertia_
    print(result.inertia_)

# Print and plot the cost for each number of clusters. The goal is to see the "elbow curve" to find out which number of clusters gives the most reasonable result
print(cost_array)
fig = pl.figure()
plt.xticks(np.arange(20))
pl.plot(cost_array[1:20])

# Since six clusters seem to be ideal, lets run the algoritm with six clusters and apply the result to the array with whiskys.
km = KMeans(nbr_clusters, init='k-means++', max_iter=100, n_init=1, verbose=True)
result = km.fit(whiskies)

print(result.inertia_)
print(result.cluster_centers_)
print(km.labels_)
print(result.labels_)

clusters = np.zeros(nbr_clusters)
fig = pl.figure()
for c in range(nbr_clusters):
    cluster_members = result.labels_ == c
    pl.plot(whiskies_pca[cluster_members, 0], whiskies_pca[cluster_members, 1], 'o')
    print("Members of cluster %d" % c)
    #print(cluster_members)
    print(whiskies_full[cluster_members, 1])

pl.show()
