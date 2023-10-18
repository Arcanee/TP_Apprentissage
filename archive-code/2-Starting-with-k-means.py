"""
Created on 2023/09/11

@author: huguet
"""
import os
os.environ["OMP_NUM_THREADS"] = '4'

import numpy as np
import matplotlib.pyplot as plt
import time
import sys

from scipy.io import arff
from sklearn import cluster
from sklearn import metrics

##################################################################
# Exemple :  k-Means Clustering

path = './artificial/'
name="R15.arff"

#path_out = './fig/'
databrut = arff.loadarff(open(path+str(name), 'r'))
datanp = np.array([[x[0],x[1]] for x in databrut[0]])

# PLOT datanp (en 2D) - / scatter plot
# Extraire chaque valeur de features pour en faire une liste
# EX : 
# - pour t1=t[:,0] --> [1, 3, 5, 7]
# - pour t2=t[:,1] --> [2, 4, 6, 8]
print("---------------------------------------")
print("Affichage données initiales            "+ str(name))
f0 = datanp[:,0] # tous les élements de la première colonne
f1 = datanp[:,1] # tous les éléments de la deuxième colonne

#plt.figure(figsize=(6, 6))
plt.scatter(f0, f1, s=8)
plt.title("Donnees initiales : "+ str(name))
#plt.savefig(path_out+"Plot-kmeans-code1-"+str(name)+"-init.jpg",bbox_inches='tight', pad_inches=0.1)
plt.show()

y_metrique = []
x_metrique = []

N = int(input("Nb max de clusters à tester :\n"))
tps1 = time.time()

for k in range (2, N+1):
    x_metrique.append(k)

    # Run clustering method for a given number of clusters
    print("------------------------------------------------------")
    print("Appel KMeans pour une valeur de k fixée")
    model = cluster.MiniBatchKMeans(n_clusters=k, init='k-means++', n_init=1)
    model.fit(datanp)
    labels = model.labels_
    # informations sur le clustering obtenu
    iteration = model.n_iter_
    inertie = model.inertia_
    centroids = model.cluster_centers_
    labels = model.labels_

    coef_silhouette = metrics.silhouette_score(datanp, labels)
    y_metrique.append(coef_silhouette)

tps2 = time.time()

plt.scatter(x_metrique, y_metrique, marker="x", color="red")
plt.plot(x_metrique, y_metrique)
plt.show()

k = np.argmax(y_metrique)+2
print("Solution optimale d'après silhouette :", k, "clusters")




# Run clustering method for best number of clusters
print("------------------------------------------------------")
print("Appel KMeans pour une valeur de k fixée")
model = cluster.KMeans(n_clusters=k, init='k-means++', n_init=1)
model.fit(datanp)
labels = model.labels_
# informations sur le clustering obtenu
iteration = model.n_iter_
inertie = model.inertia_
centroids = model.cluster_centers_
labels = model.labels_


#plt.figure(figsize=(6, 6))
plt.scatter(f0, f1, c=labels, s=8)
plt.scatter(centroids[:, 0],centroids[:, 1], marker="x", s=50, linewidths=3, color="red")
plt.title("Données après clustering : "+ str(name) + " - Nb clusters ="+ str(k))
#plt.savefig(path_out+"Plot-kmeans-code1-"+str(name)+"-cluster.jpg",bbox_inches='tight', pad_inches=0.1)
plt.show()

print("nb clusters =",k,", nb iter =",iteration, ", inertie = ",inertie, ", runtime = ", round((tps2 - tps1)*1000,2),"ms")
#print("labels", labels)

from sklearn.metrics.pairwise import euclidean_distances
dists = euclidean_distances(centroids)
print("==dists==")
print(dists)
print("==centroids==")
print(centroids)



print("\nINTRA CLUSTER\n")
#DISTANCES EXEMPLES / CENTRES

clusters = []
dmin = []
dmax = []
dmoy = []

for i in range(k):
    clusters.append([])
    dmin.append(9999)
    dmax.append(0)
    dmoy.append(0)


for p in range(len(datanp)):
    clust = labels[p]
    point = datanp[p]
    clusters[clust].append(point)
    dist = np.linalg.norm(point-centroids[clust])

    dmin[clust] = min(dmin[clust], dist)
    dmax[clust] = max(dmax[clust], dist)
    dmoy[clust] += dist

for i in range (k):
    dmoy[i] /= len(clusters[i])


print ("dist min par cluster : ", dmin)
print ("dist max par cluster : ", dmax)
print ("dist moy par cluster : ", dmoy)


# print("\nINTER CLUSTER\n")
# #DISTANCES ENTRE CLUSTERS

# dmin = []
# dmax = []
# dmoy = []
# for i in range(k):
#     dmin.append(9999)
#     dmax.append(0)
#     dmoy.append(0)


# for l in range(k):
#     for cA in clusters[l]:
#         for cB in clusters[(l+1)%k]:
#             dist = np.linalg.norm(cA-cB)
#             dmin[l] = min(dmin[l], dist)
#             dmax[l] = max(dmax[l], dist)
#             dmoy[l] += dist

#     dmoy[l] /= len(clusters[l]) * len(clusters[(l+1)%k])


# print ("dist min entre cluster : ", dmin)
# print ("dist max entre cluster : ", dmax)
# print ("dist moy entre cluster : ", dmoy)

print("\nINTER CENTRES\n")
#DISTANCES ENTRE LES CENTRES
dist_centres = []
for i in range(len(dists)):
    for j in range(i):
        dist_centres.append(dists[i][j])

print ("dist min entre les centres : ", min(dist_centres))
print ("dist max entre les centres : ", max(dist_centres))
print ("dist moy entre les centres : ", sum(dist_centres)/len(dist_centres))

coef_silhouette = metrics.silhouette_score(datanp, labels)
print ("coefficient de silhouette : ", coef_silhouette)



