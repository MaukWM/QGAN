import numpy as np
from sklearn import datasets
from sklearn import preprocessing
import matplotlib.pyplot as plt


class OneClusters:

    def __init__(self):
        pass

    def generate(self, n):
        data = datasets.make_blobs(n_samples=n * 5, n_features=2, centers=[[0.75, 0.25]], cluster_std=0.1)

        data = [[el[0], el[1]] for el in data[0] if 0 < el[0] < 1 and 0 < el[1] < 1]

        data = data[:n]


        return np.array(data)


class TwoClusters:

    def __init__(self):
        pass

    def generate(self, n):
        data = datasets.make_blobs(n_samples=n * 5, n_features=2, centers=[[0.25, 0.75], [0.75, 0.25]], cluster_std=[0.1, 0.005])

        data = [[el[0], el[1]] for el in data[0] if 0 < el[0] < 1 and 0 < el[1] < 1]

        data = data[:n]
        # X = data[0].transpose()[0]
        # Y = data[0].transpose()[1]
        #
        # norm_x = [((x - min(X)) / (max(X) - min(X))) for x in X]
        # norm_y = [((y - min(Y)) / (max(Y) - min(Y))) for y in Y]

        # return np.dstack((norm_x, norm_y))

        return np.array(data)

data = TwoClusters().generate(100).transpose()

plt.scatter(data[0], data[1])
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.show()
