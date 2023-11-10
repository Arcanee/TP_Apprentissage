import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.io import arff
from sklearn import cluster
from sklearn import metrics
from scipy.cluster.hierarchy import dendrogram


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix) #, **kwargs)



###################################################################
# Exemple : Agglomerative Clustering

linkageMethod = 'ward'
path = './artificial/'
name="jain.arff"

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



y_metrique_dist = []
x_metrique_dist = []
y_metrique_clust = []
x_metrique_clust = []


### ESTIMER la meilleure distance (5-10-15-20)
# 
tps1 = time.time()
for seuil_dist in np.arange(10,500,10):
    x_metrique_dist.append(seuil_dist)

    model = cluster.AgglomerativeClustering(distance_threshold=seuil_dist, linkage=linkageMethod, n_clusters=None)
    model = model.fit(datanp)
    labels = model.labels_

    if model.n_clusters_>1:
        coef_silhouette = metrics.silhouette_score(datanp, labels)
    else:
        coef_silhouette=0
    y_metrique_dist.append(coef_silhouette)
    
tps2 = time.time()

plt.scatter(x_metrique_dist, y_metrique_dist, marker="x", color="red")
plt.title(f"Score de silhouette selon la distance [{name}]")
plt.plot(x_metrique_dist, y_metrique_dist)
plt.show()

bestDist = x_metrique_dist[np.argmax(y_metrique_dist)]
print("Solution optimale d'après silhouette :", bestDist, "(dist)")

model = cluster.AgglomerativeClustering(distance_threshold=bestDist, linkage=linkageMethod, n_clusters=None, compute_distances=True)
model = model.fit(datanp)
labels = model.labels_
plot_dendrogram(model)
plt.show()

k = model.n_clusters_
leaves=model.n_leaves_
plt.scatter(f0, f1, c=labels, s=8)
plt.title("Clustering agglomératif ("+linkageMethod+", distance_treshold= "+str(bestDist)+") "+str(name))
plt.show()
print("nb clusters =",k,", nb feuilles = ", leaves, " runtime = ", round((tps2 - tps1)*1000,2),"ms")




###
# ESTIMER le meilleur nombre de clusters (2-3-4-5-6-7-8-9-10-11-12-13-14-15)
###
tps3 = time.time()
for n_clust in np.arange(2,21,1):
    x_metrique_clust.append(n_clust)

    model = cluster.AgglomerativeClustering(linkage=linkageMethod, n_clusters=n_clust, compute_distances=True)
    model = model.fit(datanp)
    labels = model.labels_

    coef_silhouette = metrics.silhouette_score(datanp, labels)
    y_metrique_clust.append(coef_silhouette)
    
tps4 = time.time()

plt.scatter(x_metrique_clust, y_metrique_clust, marker="x", color="red")
plt.plot(x_metrique_clust, y_metrique_clust)
plt.title(f"Score de silhouette selon le nombre de clusters [{name}]")
plt.show()

bestNClust = x_metrique_clust[np.argmax(y_metrique_clust)]
print("Solution optimale d'après silhouette :", bestNClust, "clusters")

model = cluster.AgglomerativeClustering(linkage=linkageMethod, n_clusters=bestNClust)
model = model.fit(datanp)
labels = model.labels_

k = model.n_clusters_
leaves=model.n_leaves_
plt.scatter(f0, f1, c=labels, s=8)
plt.title("Clustering agglomératif ("+linkageMethod+", n_cluster= "+str(k)+") "+str(name))
plt.show()
print("nb clusters =",bestNClust,", nb feuilles = ", leaves, " runtime = ", round((tps4 - tps3)*1000,2),"ms")

print("\n\n TPS TOTAL =", tps2-tps1+tps4-tps3, "s")