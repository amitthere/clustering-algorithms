import numpy as np
from sklearn.decomposition import PCA
import seaborn as sb
import matplotlib.pyplot as plt

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
        self.data = np.genfromtxt(tabfile, dtype = float, delimiter = '\t')

def rand_jaccard(labels, calculated):

    labelMatrix = np.zeros((labels.shape[0], labels.shape[0]))
    calculatedMatrix = np.zeros((labels.shape[0], labels.shape[0]))
    for  i in range(labels.shape[0]):
        for j in range(labels.shape[0]):
            if(np.array_equal(labels[i], labels[j])):
                labelMatrix[i][j] = 1
            if(np.array_equal(calculated[i], calculated[j])):
                calculatedMatrix[i][j] = 1
    M11 = M00 = M10 = M01 = 0
    for i in range(calculatedMatrix.shape[0]):
        for j in range(calculatedMatrix.shape[0]):
            if calculatedMatrix[i][j] == labelMatrix[i][j] == 1:
                M11 = M11 + 1
            elif calculatedMatrix[i][j] == labelMatrix[i][j] == 0:
                M00 = M00 + 1
            elif calculatedMatrix[i][j] == 1 and labelMatrix[i][j] == 0:
                M10 = M10 + 1
            elif calculatedMatrix[i][j] == 0 and labelMatrix[i][j] == 1:
                M01 = M01 + 1
    rand_index = float(M11 + M00)/float(M11 + M00 + M10 + M01)
    jaccard_coefficient = float(M11)/float(M11 + M10 + M01)
    print('Rand Coefficient = ', rand_index)
    print('Jaccard Coefficient = ', jaccard_coefficient)

def plotPCA(pcaComponents, labels):
    x = pcaComponents[:, 0]
    y = pcaComponents[:, 1]
    fig = plt.figure()
    scatter = sb.scatterplot(x, y, hue = labels)
    plot = scatter.get_figure()
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.title('Principal component analysis plot on Cho.txt with 5 clusters')
    plt.legend()
    plt.show()
    plot.savefig('../Plots/HAC_CHO_PCA.png')

def main():
    file = Import('../data/cho.txt', "TAB")