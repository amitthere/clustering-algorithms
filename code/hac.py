import numpy as np
from numpy import *
from sklearn.metrics.pairwise import euclidean_distances
import sys
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sb

class Import:
    """
    Imports data from various sources.
    Currently, just tab-delimited files are supported
    """

    def __init__(self, file, ftype):
        self.data = None
        self.prefixed_data = None
        self.file = file
        if ftype == "TAB":
            self.import_tab_file(self.file)

    def import_tab_file(self, tabfile):
        self.data = np.genfromtxt(tabfile, dtype = float, delimiter = '\t')

class ExternalIndex:

    def __init__(self, groundtruth, clusters):
        self.ground = groundtruth
        self.clusters = clusters
        self.ground_incidence_matrix = self.incidence_matrix(self.ground)
        self.cluster_incidence_matrix = self.incidence_matrix(self.clusters)
        self.M11, self.M00, self.M10, self.M01 \
            = self.categories(self.ground_incidence_matrix, self.cluster_incidence_matrix)

    def incidence_matrix(self, clusters):
        N = clusters.shape[0]
        matrix = np.zeros((N, N), dtype='int')

        for i in range(clusters.shape[0]):
            for j in range(i + 1, clusters.shape[0]):
                if (clusters[i] == clusters[j]) and (clusters[i] != -1 or clusters[j] != -1):
                    matrix[i][j] = matrix[j][i] = 1
        return matrix

    def categories(self, ground_incidence_matrix, cluster_incidence_matrix):
        M11 = M00 = M10 = M01 = 0
        for i in range(cluster_incidence_matrix.shape[0]):
            for j in range(cluster_incidence_matrix.shape[0]):
                if cluster_incidence_matrix[i][j] == ground_incidence_matrix[i][j] == 1:
                    M11 = M11 + 1
                elif cluster_incidence_matrix[i][j] == ground_incidence_matrix[i][j] == 0:
                    M00 = M00 + 1
                elif cluster_incidence_matrix[i][j] == 1 and ground_incidence_matrix[i][j] == 0:
                    M10 = M10 + 1
                elif cluster_incidence_matrix[i][j] == 0 and ground_incidence_matrix[i][j] == 1:
                    M01 = M01 + 1
        return M11, M00, M10, M01

    def rand_index(self):
        rand_index = float(self.M11 + self.M00)/float(self.M11 + self.M00 + self.M10 + self.M01)
        return rand_index

    def jaccard_coefficient(self):
        jaccard_coefficient = float(self.M11) / float(self.M11 + self.M10 + self.M01)
        return jaccard_coefficient


def eucli_dis(data):
    distance_matrix = euclidean_distances(data, data)
    return distance_matrix

def min_value(dis_r, dis_c, disMat, number_of_clusters):
    min = sys.maxsize
    x = -1
    y = -1
    for i in range(0,disMat.shape[0]):
        for j in range(0, disMat.shape[0]):
            if i !=j and min > disMat[i][j]:
                min = disMat[i][j]
                x = i
                y = j
    #print(min)
    #print(x, y)
    cluster[x].extend(cluster[y])
    del cluster[y]
    #print(cluster)
    clusters(disMat, dis_r, dis_c, x, y, number_of_clusters)

def clusters(disMat, dis_r, dis_c, x, y, number_of_clusters):
    for i in range(0,disMat.shape[0]):
        if x == i or y == i:
            continue
        if disMat[i][x] > disMat[i][y]:
            disMat[i][x] = disMat[i][y]
        if disMat[x][i] > disMat[y][i]:
            disMat[x][i] = disMat[y][i]
    disMat = np.delete(disMat, y, 0)
    disMat = np.delete(disMat, y, 1)
    if(disMat.shape != (number_of_clusters, number_of_clusters)):
        min_value(dis_r=dis_r, dis_c=dis_c, disMat=disMat, number_of_clusters=number_of_clusters)
    else:
        print("Distance Matrix")
        print(disMat)

def principal_component_analysis(data, labels):
    pca_data = PCA(n_components=2).fit_transform(data)
    plotPCA(pca_data, labels)

def plotPCA(pcaComponents, labels):
    x = pcaComponents[:, 0]
    y = pcaComponents[:, 1]
    fig = plt.figure()
    scatter = sb.scatterplot(x, y, hue = labels)
    plot = scatter.get_figure()
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.title('Principal component analysis plot on Cho.txt with 10 clusters')
    plt.legend()
    plt.show()
    #plot.savefig('../Plots/HAC_CHO_PCA.png')

def main():
    sys.setrecursionlimit(1500)
    file = Import('../data/iyer.txt', "TAB")
    number_of_clusters = 10
    file0 = file.data
    gene_id = file0[:, 0]
    final = len(gene_id)
    gene_ids = np.ravel(gene_id)
    global cluster
    cluster = []
    for i in range(len(gene_ids)):
        cluster.append([int(file0[i][0]) - 1])
    # print(clusters)
    ground_truth_label = file0[:,1]
    gene_data = file0[:,2:]
    rows, columns = gene_data.shape
    #print(gene_data.shape)
    disMat = eucli_dis(gene_data)
    #print(disMat.shape)
    dis_r, dis_c = disMat.shape
    min_value(dis_r, dis_c, disMat, number_of_clusters)
    print(cluster)
    list = np.zeros(len(gene_id), dtype=int)
    for i in range(len(cluster)):
        for j in cluster[i]:
            list[j] = i
    #print(list)
    length = len(list)
    #print(length)
    rand = ExternalIndex(ground_truth_label, list)
    jaccard = ExternalIndex(ground_truth_label, list)
    print('Rand index= ', rand.rand_index())
    print('Jaccard coefficient= ', jaccard.jaccard_coefficient())
    principal_component_analysis(gene_data, list)

if __name__ == "__main__":
    main()