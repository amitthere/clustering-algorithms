import numpy as np
from numpy import *
from sklearn.metrics.pairwise import euclidean_distances
import sys

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


file = Import('../data/cho.txt', "TAB")
number_of_clusters = 3
file0 = file.data
gene_id = file0[:,0]
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
        #disMat[i][x] = min(disMat[i][x], disMat[i][y])
        #disMat[x][i] = min(disMat[x][i], disMat[y][i])
    disMat = np.delete(disMat, y, 0)
    disMat = np.delete(disMat, y, 1)
    if(disMat.shape != (number_of_clusters, number_of_clusters)):
        min_value(dis_r=dis_r, dis_c=dis_c, disMat=disMat, number_of_clusters=number_of_clusters)
    else:
        print(disMat)

def main():

    #print(gene_ids)
    ground_truth_label = file0[:,1]
    gene_data = file0[:,2:]
    rows, columns = gene_data.shape
    #print(gene_data.shape)
    truth_matrix = np.zeros((rows,rows))
    disMat = eucli_dis(gene_data)
    print(disMat.shape)
    dis_r, dis_c = disMat.shape
    min_value(dis_r, dis_c, disMat, number_of_clusters)

if __name__ == "__main__":
    main()