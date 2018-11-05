import numpy as np
import configparser
from MapReduceDriver import MapReduceKMeans
from visualization import Visualization
from clustervalidation import ExternalIndex


def main():
    config = configparser.ConfigParser()
    config.read(r'config.ini')
    clusters = int(config['DEFAULT']['ClusterCount'])
    file = open(config['DEFAULT']['InputDirectory']+'/'+config['DEFAULT']['InputFile'], 'r')
    fdata = np.genfromtxt(file, dtype=float, delimiter='\t')

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
