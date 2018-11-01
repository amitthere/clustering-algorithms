#!/usr/bin/python3

import os
import sys
import numpy as np


class Reducer:

    def __init__(self):
        self.data = None
        self.clusters = None
        self.centroids = None

    def compute_centroids(self, clusters, data):
        ClustersCount = int(os.environ['ClustersCount'])
        new_centroids = np.zeros((ClustersCount, data.shape[1]), dtype=float)

        for i in range(new_centroids.shape[0]):
            idx = np.where(clusters == i)
            new_centroids[i] = np.mean(data[idx[0]], axis=0)
        return new_centroids

    def emit_output(self):
        tc = np.zeros((self.centroids.shape[0], 2 + self.centroids.shape[1]), dtype=float)
        cc = np.arange(self.centroids.shape[0], dtype=int)
        tc[:, 0] = cc
        tc[:, 1] = -1
        tc[:, 2:] = self.centroids

        for i in tc:
            print(str(i[0]) + '\t' + str(i[1]) + '\t' + ','.join(str(x) for x in i[2:]))
        return

    def reduce(self):
        data_list = []

        for line in sys.stdin:
            line = line.strip()
            cluster, id, serialized_row = line.split('\t')
            datapoint = np.fromstring(serialized_row, dtype='float', sep=',')

            row = np.zeros((1, 2 + datapoint.shape[0]), dtype=float)
            row[0][0] = int(cluster)
            row[0][1] = int(id)
            row[0][2:] = datapoint
            data_list.append(row)

        data = np.vstack(tuple(data_list))
        centroids = self.compute_centroids(data[:,0], data[:, 2:])

        self.data = data
        self.clusters = data[:, 0]
        self.centroids = centroids

        self.emit_output(centroids)
        return


def main():
    reducer = Reducer()
    reducer.reduce()
    pass


if __name__ == "__main__":
    main()
