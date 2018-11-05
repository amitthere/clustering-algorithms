import os
import subprocess
import numpy as np


class MapReduceKMeans:
    """
    Map Reduce K-Means clustering implementation
    """

    def __init__(self, data, ground_truth, cluster_count, mr_inputfile_name, streaming_jar,
                 mapper, reducer, hdfs_input_dir, hdfs_output_dir, tmp_dir):
        self.data = data
        self.centroids = None
        self.ClusterCount = cluster_count
        self.clusters = np.random.randint(cluster_count, size=(data.shape[0], 1))
        self.ground_truth_clusters = ground_truth.astype('int')
        self.mr_inputfile_name = mr_inputfile_name
        self.streaming_jar = streaming_jar
        self.mapper = mapper
        self.reducer = reducer
        self.hdfs_input_dir = hdfs_input_dir
        self.hdfs_output_dir = hdfs_output_dir
        self.temporary_directory = tmp_dir

    def initial_random_centroids(self, num):
        """
        Random centroids chosen from the input dataset
        :param num: number of centroids
        :return centroid_indices: indices of chosen centroids
        """
        centroid_indices = np.random.choice(self.data.shape[0], num)
        self.centroids = self.data[centroid_indices]
        return centroid_indices

    def initial_centroids(self, indices):
        """
        Set choosen initial centroids
        :param indices: indices of chosen centroids
        :return indices: indices of chosen centroids
        """
        self.centroids = self.data[indices]
        return indices

    def centroid_distance(self, centroid):
        """
        Euclidean distance implemented to calculate distance of each object in dataset from given centroid
        :param centroid: one centroid
        :return: distance of each object in the dataset from input centroid
        """
        return np.sqrt(np.sum(np.square(self.data - centroid), axis=1))

    def distance_from_centroids(self):
        rows = self.data.shape[0]
        cols = self.centroids.shape[0]
        distance_matrix = np.zeros((rows, cols), dtype=float)

        for index, centroid in enumerate(self.centroids):
            distance_matrix[:, index] = self.centroid_distance(centroid)

        return distance_matrix

    def assign_clusters(self, distance_matrix):
        """
        Assign Objects to clusters with minimum distance to centroid
        :param distance_matrix: NxC with distance of each object from C centroids
        :return: list of clusters
        """
        clusters = np.argmin(distance_matrix, axis=1)
        return clusters

    def map_reduce(self):

        mapsandreduces = ' -D mapreduce.job.maps=1 -D mapreduce.job.reduces=1 '

        job = 'hadoop jar ' + self.streaming_jar + mapsandreduces + ' -file ' + self.mapper + ' -mapper ' + self.mapper \
              + ' -file ' + self.reducer + ' -reducer ' + self.reducer + ' -cmdenv ClusterCount=' + \
              str(self.ClusterCount) + ' -input ' + self.hdfs_input_dir + ' -output ' + self.hdfs_output_dir

        os.system(job)
        return

    def place_input_file(self):
        input_file = np.vstack((self.data[:, 2:], self.centroids[:, 2:]))
        np.savetxt(self.mr_inputfile_name, input_file, delimiter='\t')
        os.system('hadoop fs -rm ' + self.hdfs_input_dir + self.mr_inputfile_name)
        os.system('hadoop fs -put ' + self.mr_inputfile_name + ' ' + self.hdfs_input_dir)
        os.system('hadoop fs -rm -R ' + self.hdfs_output_dir)
        return

    def read_mr_output(self):
        cat = subprocess.Popen(['hadoop', 'fs', '-cat', self.hdfs_output_dir + 'part*'], stdout=subprocess.PIPE)
        centroids = np.zeros(self.centroids.shape, dtype=float)

        for i, line in enumerate(cat.stdout):
            cluster, id, cpoint = line.split('\t')
            centroids[i][0] = float(cluster)
            centroids[i][1] = float(id)
            centroids[i][2:] = [float(i) for i in cpoint.split(',')]

        return centroids

    def kmeans(self, total_iters=-1):
        """
        K Means Clustering implementation on Hadoop Streaming
        :param total_iters: count of iterations before stopping. -1 for unlimited
        :return:
        """
        iteration = 1
        prev_centroids = self.centroids
        while True:

            self.place_input_file()

            # run one iteration of Map Reduce
            self.map_reduce()

            self.centroids = self.read_mr_output()

            # assign mrkm.clusters
            if np.array_equal(prev_centroids, self.centroids) or total_iters == iteration:
                break
            else:
                prev_centroids = self.centroids
                iteration = iteration + 1

        # Algorithm has converged
        self.clusters = self.assign_clusters(self.distance_from_centroids())
        return
