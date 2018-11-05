import numpy as np
import matplotlib
matplotlib.use('Agg')
from lib.clustervalidation import ExternalIndex, Visualization
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import time


class Kmeans:
    """
    K-Means clustering implementation
    """

    def __init__(self, data, ground_truth, cluster_count):
        self.init_centroids = None
        self.centroids = None
        self.data = data
        self.cluster_count = cluster_count
        self.clusters = np.zeros((self.data.shape[0], 1), dtype=int)
        self.ground_truth_clusters = ground_truth.astype('int')

    def initial_random_centroids(self, num):
        """
        Random centroids chosen from the input dataset
        :param num:
        :return centroid_indices:
        """
        centroid_indices = np.random.choice(self.data.shape[0], num)
        self.init_centroids = self.data[centroid_indices]
        self.centroids = self.init_centroids
        return centroid_indices

    def initial_centroids(self, *args):
        point_indices = [i-1 for i in args]
        self.init_centroids = self.data[point_indices]
        self.centroids = self.init_centroids
        return point_indices

    def euclidean_distance(self, objSet, obj):
        """Euclidean distance between set of objects and single object"""
        return np.linalg.norm(objSet - obj, ord=2, axis=1)

    def centroid_distance(self, centroid):
        """ Euclidean distance implemented to calculate distance of each object in dataset from given centroid """
        return np.sqrt(np.sum(np.square(self.data - centroid), axis=1))

    def distance_from_centroids(self):
        rows = self.data.shape[0]
        cols = self.centroids.shape[0]
        distance_matrix = np.zeros((rows, cols), dtype=float)

        for index, centroid in enumerate(self.centroids):
            distance_matrix[:, index] = self.centroid_distance(centroid)

        return distance_matrix

    def assign_clusters(self, distance_matrix):
        """Assign Objects to clusters with minimum distance to centroid"""
        clusters = np.argmin(distance_matrix, axis=1)
        return clusters

    def compute_centroids(self, clusters):
        """Compute new centroids from objects assigned to clusters"""
        new_centroids = np.zeros(self.centroids.shape, dtype=float)
        # horizontally_merged_array = np.hstack((1D_array[:, np.newaxis], 2D_array))
        for i in range(new_centroids.shape[0]):
            idx = np.where(clusters == i)
            new_centroids[i] = np.mean(self.data[idx[0]], axis=0)
        return new_centroids

    def kmeans_algorithm(self, iterations = None, log=False):
        """
        Performs K-Means clustering based on set initial centroids
        :param log: set to True if 2D plot of each iteration is required
        :return:
        """
        # make sure initial centroids are set
        if self.centroids.all() == None:
            self.initial_random_centroids(self.cluster_count)

        if log:
            pca = PCA(n_components=2)
            pca.fit(self.data.T)

        i = 0
        while True:
            if log:
                self.log_iterations_of_kmeans(pca.components_, self.clusters)

            distance_matrix = self.distance_from_centroids()
            clusters = self.assign_clusters(distance_matrix)
            new_centroids = self.compute_centroids(clusters)

            # check if clusters are same, NOT the centroids
            if np.array_equal(self.clusters, clusters):
                break
            else:
                self.centroids = new_centroids
                self.clusters = clusters
            i = i + 1
            if i == iterations:
                break

        self.clusters = self.clusters + 1
        return

    def log_iterations_of_kmeans(self, components, clusters):
        if clusters.ndim == 1:
            clusters = clusters[:, None]
            clusters = clusters.T
        else:
            clusters = clusters.T
        centroids = np.zeros((2,5), dtype=float)
        centroid_cluster = np.ones((1,5), dtype=int)
        centroid_cluster.fill(6)
        for i in range(centroids.shape[1]):
            idx = np.where(clusters == i)
            centroids[:, i] = np.mean(components[:, idx[0]], axis=1)
        points = np.hstack((components, centroids))
        categories = np.hstack((clusters, centroid_cluster))
        matplotlib.use('Agg')  # forces python to not use xwindow to display the plot in separate window
        # plt.figure(num=None, figsize=(15, 12), dpi=80, facecolor='w', edgecolor='k')
        plt.scatter(points[0], points[1], c=np.squeeze(categories), edgecolors='black', cmap='tab10')
        # plt.grid()
        plt.savefig(r'../log/k-means-iteration-'+str(time.time())+'.jpg')
        plt.close()
        return
