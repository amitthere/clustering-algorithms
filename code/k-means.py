import numpy as np

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
        self.data = np.genfromtxt(tabfile, dtype=str, delimiter='\t')


class Kmeans:
    """
    K-Means clustering implementation
    """

    def __init__(self):
        self.centroid = None




def main():
    dataset1 = Import(r'../data/cho.txt', 'TAB')
    dataset2 = Import(r'../data/iyer.txt', 'TAB')


    return


if __name__ == "__main__":
    main()
