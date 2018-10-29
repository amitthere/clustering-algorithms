
import os
import sys
import numpy as np


class Map:

    def __init__(self, data, centroids):
        self.data = data
        self.centroids = centroids

    def centroid_distance(self, point, centroid):
        """ Euclidean distance implemented to calculate distance of each object in dataset from given centroid """
        return np.sqrt(np.sum(np.square(point - centroid), axis=1))

    def distance_from_centroids(self, point):
        rows = 1
        cols = self.centroids.shape[0]
        distance_matrix = np.zeros((rows, cols), dtype=float)

        for index, centroid in enumerate(self.centroids):
            distance_matrix[:, index] = self.centroid_distance(point, centroid)

        return distance_matrix

    def assign_clusters(self, distance_matrix):
        """Assign Objects to clusters with minimum distance to centroid"""
        clusters = np.argmin(distance_matrix, axis=1)
        return clusters

    def map(self):
        for id, row in enumerate(self.data):
            serialized_row = ','.join(str(x) for x in row)
            cluster = self.assign_clusters(self.distance_from_centroids(row))[0]
            print(str(cluster) + '\t' + str(id) + '\t' + serialized_row)
        return


def main():
    input = np.genfromtxt(sys.stdin, dtype='float')
    clusters = int(os.environ['ClustersCount'])
    data = input[:input.shape[0] - clusters, :]
    centroids = input[input.shape[0] - clusters:, :]
    mapper = Map(data, centroids)
    mapper.map()
    return


if __name__ == "__main__":
    main()
