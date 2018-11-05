import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


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
        classes = ['Cluster-' + str(i) for i in np.unique(self.clusters)]
        #plt.legend(loc=4)
        plt.scatter(components[0], components[1], c=self.clusters, edgecolors='black')
        plt.grid()
        plt.savefig(filename)
