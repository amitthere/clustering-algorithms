import numpy as np
from kmeans import Kmeans
from visualization import Visualization
from clustervalidation import ExternalIndex


def main():
    dataset1 = np.genfromtxt(r'../data/new_dataset_1.txt', dtype=float, delimiter='\t')
    dataset2 = np.genfromtxt(r'../data/cho.txt', dtype=float, delimiter='\t')

    km1 = Kmeans(dataset1[:, 2:], dataset1[:, 1], 3)
    km2 = Kmeans(dataset2[:, 2:], dataset2[:, 1], 10)

    ic1 = km1.initial_centroids(3, 5, 9)
    #ic1 = km1.initial_random_centroids(5)
    ic2 = km2.initial_random_centroids(5)
    # km1.centroids = km1.init_centroids = np.loadtxt(r'../log/cho_ground_centroids.txt')

    # specify iteration as parameter here
    km1.kmeans_algorithm()
    km2.kmeans_algorithm()

    extr_index_validation1 = ExternalIndex(km1.ground_truth_clusters, km1.clusters)
    extr_index_validation2 = ExternalIndex(km2.ground_truth_clusters, km2.clusters)

    print('Rand Index of dataset1 clusters :', extr_index_validation1.rand_index())
    print('Jaccard Coefficient of dataset1 clusters :', extr_index_validation1.jaccard_coefficient())

    print('Rand Index of dataset2 clusters :', extr_index_validation2.rand_index())
    print('Jaccard Coefficient of dataset2 dataset clusters :', extr_index_validation2.jaccard_coefficient())

    plot1 = Visualization(dataset1.data[:, 2:], km1.clusters, dataset1.data[:, 1])
    plot2 = Visualization(dataset2.data[:, 2:], km2.clusters, dataset2.data[:, 1])
    plot1.plot(r'../log/td1.jpg')
    plot2.plot(r'../log/cho2.jpg')

    # gene_cluster_matched = km1.cluster_validation()
    # print('Genes that matched in clusters: ', gene_cluster_matched)

    return


if __name__ == "__main__":
    main()
