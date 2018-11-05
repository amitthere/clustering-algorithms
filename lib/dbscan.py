import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from clustervalidation import ExternalIndex
from sklearn.metrics.pairwise import euclidean_distances


class DBSCAN:
    """
    Density-Based Spatial Clustering of Applications with Noise (DBSCAN)
    """

    def __init__(self):
        pass

def eucli_dis(data):
    distance_matrix = euclidean_distances(data, data)
    return distance_matrix


def plotPCA(pcaComponents, labels):
    x = pcaComponents[:, 0]
    y = pcaComponents[:, 1]
    fig = plt.figure()
    scatter = sb.scatterplot(x, y, hue=labels)
    plot = scatter.get_figure()
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.title('Principal component analysis plot on Iyer.txt with Epsilon = 1.13 and minPts = 3')
    plt.legend()
    plt.show()
    plot.savefig('../Plots/DBSCAN_IYER_PCA.png')


def principal_component_analysis(data, labels):
    pca_data = PCA(n_components=2).fit_transform(data)
    plotPCA(pca_data, labels)


def dbscan(gene_data, rows, disMat):
    cluster = np.zeros(rows)
    clusterid = 0
    for pt1 in range(rows):
        if cluster[pt1] != 0:
            continue
        neighbours = regionQuery(gene_data, pt1, rows, disMat)
        if len(neighbours) < minPts:
            cluster[pt1] = -1
            continue
        clusterid = clusterid + 1
        expandCluster(pt1, neighbours, clusterid, cluster)
        unique, counts = np.unique(cluster, return_counts=True)
        unique = [int(i) for i in unique]
        #print("Cluster: count = ", str(dict(zip(unique, counts))))
        centroids = {}
        for k in range(int(np.min(cluster)), int(np.max(cluster))):
            if k == 0:
                continue
            centroids[k] = np.asarray(np.where(cluster == k)) + 1
        #print("Cluster: points in cluster = ")
        #for key, value in centroids.items():
            #print(str(key), str(value))
    return cluster


def regionQuery(gene_data, pt1, rows, disMat):
    neighbors = []
    for pt2 in range(rows):
        if disMat[pt2][pt1] < eps:
            neighbors.append(pt2)
    return neighbors


def expandCluster(pt1, neighbours, clusterid, cluster):
    cluster[pt1] = clusterid
    for i in neighbours:
        if cluster[i] == -1:
            cluster[i] = clusterid
        if cluster[i] == 0:
            cluster[i] = clusterid
            neighbor1 = regionQuery(gene_data, i, rows, disMat)
            if len(neighbor1) >= minPts:
                neighbours += neighbor1
    return


def main():
    file = np.genfromtxt(r'../data/new_dataset_1.txt', dtype=float, delimiter='\t')
    data = file.data
    global gene_data
    gene_data = data[:,2:]
    ground_truth_label = data[:,1]
    global rows
    rows, columns = gene_data.shape
    global eps
    eps = 1.2
    global minPts
    minPts = 3
    global disMat
    disMat = eucli_dis(gene_data)
    cluster = dbscan(gene_data, rows, disMat)
    #print(cluster)
    lbl = []
    for item in cluster:
        if item not in lbl:
            lbl.append(item)
    #print(lbl)
    principal_component_analysis(gene_data, cluster)
    rand = ExternalIndex(ground_truth_label, cluster)
    jaccard = ExternalIndex(ground_truth_label, cluster)
    print('Rand index= ', rand.rand_index())
    print('Jaccard coefficient= ', jaccard.jaccard_coefficient())


if __name__ == "__main__":
    main()
