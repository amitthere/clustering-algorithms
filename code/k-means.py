import numpy as np

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
        self.data = np.genfromtxt(tabfile, dtype=str, delimiter='\t')


class Kmeans:
    """
    K-Means clustering implementation
    """

    def __init__(self, data):
        self.init_centroids = None
        self.centroids = None
        self.data = data

    def initial_random_centroids(self, num):
        """
        Random centroids chosen from the input dataset
        :param num:
        :return centroid_indics:
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
        diff = self.data - centroid
        square = np.square(diff)
        sum = np.sum(square, axis=1)
        sqrt = np.sqrt(sum)
        return sqrt

    def distance_from_centroids(self):
        rows = self.centroids.shape[0]
        cols = self.centroids.shape[0]
        dist = np.zeros((rows, cols), dtype=float)

        for centroid in self.centroids:
            dist[0] = self.centroid_distance(centroid)




def main():
    dataset1 = Import(r'../data/cho.txt', 'TAB')
    dataset2 = Import(r'../data/iyer.txt', 'TAB')


    return


if __name__ == "__main__":
    main()
