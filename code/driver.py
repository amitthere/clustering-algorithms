
import os
import numpy as np
import configparser
import map, reduce


class Import:
    """
    reference - https://www.michael-noll.com/tutorials/writing-an-hadoop-mapreduce-program-in-python/
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
    def __init__(self, data, ground_truth, clusters):
        self.data = data
        self.init_centroids = None
        self.centroids = None
        self.clusters = np.random.randint(clusters, size=(data.shape[0], 1))
        self.ground_truth_clusters = ground_truth.astype('int')

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

    def map_reduce_cmd(self, mapper, reducer, hdfs_input, hdfs_output):
        streaming_cmd = 'hadoop jar /home/hadoop/hadoop/share/hadoop/tools/lib/hadoop-streaming-2.6.4.jar'
        mapper = '/home/hadoop/Documents/601/map.py'
        reducer = '/home/hadoop/Documents/601/reduce.py'
        hdfs_input = '/KMInput'
        hdfs_output = '/KMOutput'
        job = streaming_cmd + ' -mapper ' + mapper + ' -reducer ' + reducer + ' -input ' + hdfs_input + ' -output ' + hdfs_output
        return job

    def kmeans(self):

        prev_clusters = self.clusters
        while True:

            # run one iteration of MR
            os.system(self.map_reduce_cmd('', '', '', ''))

            # assign mrkm.clusters
            if np.array_equal(prev_clusters, self.clusters):
                break
            else:
                prev_clusters = self.clusters

        return


def main():
    config = configparser.ConfigParser()
    config.read(r'config.ini')
    clusters = int(config['DEFAULT']['ClusterCount'])
    file = open(config['DEFAULT']['InputDirectory']+'/'+config['DEFAULT']['InputFile'], 'r')
    fdata = Import(file, 'TAB')

    mrkm = MapReduceKMeans(fdata.data[:, 2:], fdata.data[:, 1], clusters)
    if config['DEFAULT']['Random'] == 'True':
        mrkm.initial_random_centroids(config['DEFAULT']['ClusterCount'])
    else:
        indices = config['DEFAULT']['Centroids']
        mrkm.initial_centroids([int(i) for i in indices])

    mrkm.kmeans()



if __name__ == "__main__":
    main()
