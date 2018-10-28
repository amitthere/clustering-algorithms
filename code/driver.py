
import numpy as np
import configparser
import map, reduce


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


class MapReduceKMeans:
    """
    Map Reduce K-Means clustering implementation
    """
    def __init__(self, data):
        self.data = data
        self.init_centroids = None
        self.centroids = None

    def initial_random_centroids(self, num):
        """
        Random centroids chosen from the input dataset
        :param num: number of centroids
        :return centroid_indices: indices of chosen centroids
        """
        centroid_indices = np.random.choice(self.data.shape[0], num)
        self.init_centroids = self.data[centroid_indices]
        self.centroids = self.init_centroids
        return centroid_indices

    def initial_centroids(self, indices):
        """
        Set choosen initial centroids
        :param indices: indices of chosen centroids
        :return indices: indices of chosen centroids
        """
        self.init_centroids = self.data[indices]
        self.centroids = self.init_centroids
        return indices


def main():
    config = configparser.ConfigParser()
    config.read(r'config.ini')
    file = open(config['DEFAULT']['InputDirectory']+'/'+config['DEFAULT']['InputFile'], 'r')
    data = Import(file, 'TAB')

    mrkm = MapReduceKMeans(data)
    if config['DEFAULT']['Random'] == 'True':
        mrkm.initial_random_centroids(config['DEFAULT']['ClusterCount'])
    else:
        indices = config['DEFAULT']['Centroids']
        mrkm.initial_centroids([int(i) for i in indices])


if __name__ == "__main__":
    main()
