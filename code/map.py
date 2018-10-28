
import os
import sys
import numpy as np


class Map:

    def __init__(self, data, centroids):
        self.data = data
        self.centroids = centroids

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

    def map(self):
        pass

    def combine(self):
        pass


def main():
    input = np.genfromtxt(sys.stdin, dtype='float')
    clusters = os.environ['ClustersCount']
    data = input[input.shape[0] - clusters, :]
    centroids = input[:-clusters, :]
    mapper = Map(data, centroids)
    mapper.map()
    return


if __name__ == "__main__":
    main()
