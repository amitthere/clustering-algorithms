import numpy as np
import configparser
from mrkmeans.MapReduceDriver import MapReduceKMeans
from visualization import Visualization
from clustervalidation import ExternalIndex


def main():
    config = configparser.ConfigParser()
    config.read(r'config.ini')

    cluster_count = int(config['KMEANS']['ClusterCount'])

    file = open(config['DATASET']['InputDirectory'] + config['DATASET']['InputFile'], 'r')
    fdata = np.genfromtxt(file, dtype=float, delimiter='\t')

    mapreduce_inputfile_name = config['HADOOP']['mapreduce_inputfile_name']
    streaming_jar = config['HADOOP']['StreamingJar']
    mapper = config['HADOOP']['mapper']
    reducer = config['HADOOP']['reducer']
    hdfs_input_dir = config['HADOOP']['hdfs_input_directory']
    hdfs_output_dir = config['HADOOP']['hdfs_output_directory']
    tmp_dir = config['HADOOP']['temporary_directory']

    mrkm = MapReduceKMeans(fdata.data[:, 2:], fdata.data[:, 1], cluster_count, mapreduce_inputfile_name, streaming_jar
                           , mapper, reducer, hdfs_input_dir, hdfs_output_dir, tmp_dir)
    if config['KMEANS']['Random'] == 'True':
        mrkm.initial_random_centroids(config['DEFAULT']['ClusterCount'])
    else:
        indices = config['KMEANS']['Centroids']
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
