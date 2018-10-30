
import os
import subprocess
import numpy as np
import configparser
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


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
        classes = ['Cluster-' + str(i) for i in np.unique(self.clusters)]
        plt.scatter(components[0], components[1], c=self.clusters, edgecolors='black')
        plt.grid()
        plt.savefig(filename)


class MapReduceKMeans:
    """
    Map Reduce K-Means clustering implementation
    """
    def __init__(self, data, ground_truth, clusters, hdfs_input_file):
        self.hdfs_input_file = hdfs_input_file
        self.data = data
        self.init_centroids = None
        self.centroids = None
        self.ClusterCount = clusters
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

    def map_reduce_cmd(self, mapper, reducer, hdfs_input, hdfs_output):
        streaming_cmd = 'hadoop jar /home/hadoop/hadoop/share/hadoop/tools/lib/hadoop-streaming-2.6.4.jar'
        mapsandreduces = ' -D mapreduce.job.maps=1 -D mapreduce.job.reduces=1 '
        mapper = '/home/hadoop/Documents/601/mapper.py'
        reducer = '/home/hadoop/Documents/601/reducer.py'
        hdfs_input = '/KMInput'
        hdfs_output = '/KMOutput'
        job = streaming_cmd + mapsandreduces + ' -file ' + mapper + ' -mapper ' + mapper + ' -file ' \
              + reducer +' -reducer ' + reducer + ' -cmdenv ClusterCount=' + str(self.ClusterCount) + \
              ' -input ' + hdfs_input + ' -output ' + hdfs_output
        return job

    def place_input_file(self):
        input_file = np.vstack((self.data[:, 2:], self.centroids[:, 2:]))
        np.savetxt(self.hdfs_input_file, input_file, delimiter='\t')
        os.system('hadoop fs -rm /KMInput/'+self.hdfs_input_file)
        os.system('hadoop fs -put ' + self.hdfs_input_file + ' /KMInput')
        os.system('hadoop fs -rm -R /KMOutput')
        return

    def read_mr_output(self):
        # refernce -https://stackoverflow.com/questions/12485718
        cat = subprocess.Popen(["hadoop", "fs", "-cat", "/KMOutput/part*"], stdout=subprocess.PIPE)
        centroids = np.zeros(self.centroids.shape, dtype=float)
        for i, line in enumerate(cat.stdout):
            cluster, id, cpoint = line.split('\t')
            centroids[i][0] = float(cluster)
            centroids[i][1] = float(id)
            centroids[i][2:] = [float(i) for i in cpoint.split(',')]

        return centroids

    def kmeans(self):

        prev_centroids = self.centroids
        while True:

            self.place_input_file()

            # run one iteration of MR
            os.system(self.map_reduce_cmd('', '', '', ''))

            self.centroids = self.read_mr_output()
            # assign mrkm.clusters
            if np.array_equal(prev_centroids, self.centroids):
                break
            else:
                prev_centroids = self.centroids

        # Algorithm has converged
        self.clusters = self.assign_clusters(self.distance_from_centroids())
        return


def main():
    config = configparser.ConfigParser()
    config.read(r'config.ini')
    clusters = int(config['DEFAULT']['ClusterCount'])
    file = open(config['DEFAULT']['InputDirectory']+'/'+config['DEFAULT']['InputFile'], 'r')
    fdata = Import(file, 'TAB')

    mrkm = MapReduceKMeans(fdata.data[:, 2:], fdata.data[:, 1], clusters, 'hdfs_in_file.txt')
    if config['DEFAULT']['Random'] == 'True':
        mrkm.initial_random_centroids(config['DEFAULT']['ClusterCount'])
    else:
        indices = config['DEFAULT']['Centroids']
        mrkm.initial_centroids([int(i) for i in indices.split(',')])

    mrkm.kmeans()

    ei = ExternalIndex(mrkm.ground_truth_clusters, mrkm.clusters)
    print('Rand Index : ', ei.rand_index())
    print('Jaccard Coefficient : ', ei.jaccard_coefficient())

    visual = Visualization(mrkm.data, mrkm.clusters, mrkm.ground_truth_clusters)
    visual.plot('demo.jpg')

    return


if __name__ == "__main__":
    main()
