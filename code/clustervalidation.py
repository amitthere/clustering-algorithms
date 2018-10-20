import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


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


class Visualization:

    def __init__(self, data, clusters, groundtruth):
        self.data = data
        self.clusters = clusters
        self.ground = groundtruth

    def principal_component_analysis(self, dimensions):
        pca = PCA(n_components=dimensions)
        pca.fit(self.data.T)
        return pca.components_

    def plot(self, filename):
        components = self.principal_component_analysis(2)
        plt.scatter(components[0], components[1], c=self.clusters, edgecolors='black')
        plt.grid()
        plt.savefig(filename)

