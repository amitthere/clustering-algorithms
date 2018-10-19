import numpy as np
from clustervalidation import ExternalIndex, Visualization

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
        self.data = np.genfromtxt(tabfile, dtype=float, delimiter='\t')


class Kmeans:
    """
    K-Means clustering implementation
    """

    def __init__(self, data, ground_truth):
        self.init_centroids = None
        self.centroids = None
        self.data = data
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

    def kmeans_algorithm(self):

        # make sure initial centroids are set

        while True:
            distance_matrix = self.distance_from_centroids()
            clusters = self.assign_clusters(distance_matrix)
            new_centroids = self.compute_centroids(clusters)

            # check if clusters are same, NOT the centroids
            if np.array_equal(self.clusters, clusters):
                break
            else:
                self.centroids = new_centroids
                self.clusters = clusters

        self.clusters = self.clusters + 1
        return

    # def cluster_validation(self):
    #     """Returns the number of objects matching the ground truth clusters"""
    #     clusters = self.clusters + 1
    #     ground_truth_clusters = self.ground_truth_clusters
    #     return np.sum(clusters == ground_truth_clusters)


def main():
    dataset1 = Import(r'../data/cho.txt', 'TAB')
    dataset2 = Import(r'../data/iyer.txt', 'TAB')

    km1 = Kmeans(dataset1.data[:, 2:], dataset1.data[:, 1])
    # ic1 = km1.initial_random_centroids(5)

    km1.centroids = km1.init_centroids = np.loadtxt(r'../log/cho_ground_centroids.txt')

    km1.kmeans_algorithm()

    extr_index_valid = ExternalIndex(km1.ground_truth_clusters, km1.clusters)
    print('Rand Index on Cho dataset clusters :', extr_index_valid.rand_index())
    print('Jaccard Coefficient on Cho dataset clusters :', extr_index_valid.jaccard_coefficient())


    # gene_cluster_matched = km1.cluster_validation()
    # print('Genes that matched in clusters: ', gene_cluster_matched)

    return


if __name__ == "__main__":
    main()
