# Clustering Algorithms

3 clustering algorithms implemented and tested on 4 datasets.

Parallel Implementation of K Means on Hadoop Streaming.

## 1. K-Means Clustering

## 2. Hierarchical Agglomerative Clustering

## 3. Density Based Clustering

## 4. Parallel K-Means (Implemented using Hadoop Streaming)


### Test Datasets

|     Dataset    |  Objects  | Number of Clusters |
|:--------------:|:---------:|:------------------:|
| cho            |    386    |         5          |
| iyer           |    517    |         10         |
| demo-dataset-1 |    150    |         3          |
| demo-dataset-2 |     6     |         2          |


### Dataset Format

Each row represents a gene:
1. First column is gene_id.
2. Second column is the ground truth cluster.  "-1" represents outliers.
3. Rest of the columns represent gene's expression values (attributes).


