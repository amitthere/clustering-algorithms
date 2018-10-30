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

sys.setrecursionlimit(1500)
file = Import('../data/cho.txt', "TAB")
number_of_clusters = 5
file0 = file.data
gene_id = file0[:,0]
final = len(gene_id)
gene_ids = np.ravel(gene_id)
cluster = []
for i in range(len(gene_ids)):
    cluster.append([int(file0[i][0])-1])
#print(clusters)



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

def rand_jaccard(labels, calculated):

    labelMatrix = np.zeros((labels.shape[0], labels.shape[0]))
    calculatedMatrix = np.zeros((labels.shape[0], labels.shape[0]))
    for  i in range(labels.shape[0]):
        for j in range(labels.shape[0]):
            if(np.array_equal(labels[i], labels[j])):
                labelMatrix[i][j] = 1
            if(np.array_equal(calculated[i], calculated[j])):
                calculatedMatrix[i][j] = 1
    M11 = M00 = M10 = M01 = 0
    for i in range(calculatedMatrix.shape[0]):
        for j in range(calculatedMatrix.shape[0]):
            if calculatedMatrix[i][j] == labelMatrix[i][j] == 1:
                M11 = M11 + 1
            elif calculatedMatrix[i][j] == labelMatrix[i][j] == 0:
                M00 = M00 + 1
            elif calculatedMatrix[i][j] == 1 and labelMatrix[i][j] == 0:
                M10 = M10 + 1
            elif calculatedMatrix[i][j] == 0 and labelMatrix[i][j] == 1:
                M01 = M01 + 1
    rand_index = float(M11 + M00)/float(M11 + M00 + M10 + M01)
    jaccard_coefficient = float(M11)/float(M11 + M10 + M01)
    print('Rand Coefficient = ', rand_index)
    print('Jaccard Coefficient = ', jaccard_coefficient)


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
    plt.title('Principal component analysis plot on Cho.txt with 5 clusters')
    plt.legend()
    plt.show()
    plot.savefig('../Plots/HAC_CHO_PCA.png')

def main():
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
    rand_jaccard(ground_truth_label, list)
    principal_component_analysis(gene_data, list)

if __name__ == "__main__":
    main()