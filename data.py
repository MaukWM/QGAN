from random import randrange

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
        data = datasets.make_blobs(n_samples=n * 5, n_features=2, centers=[[0.25, 0.75], [0.75, 0.25]], cluster_std=[0.05, 0.05])

        data = [[el[0], el[1]] for el in data[0] if 0 < el[0] < 1 and 0 < el[1] < 1]

        data = data[:n]
        # X = data[0].transpose()[0]
        # Y = data[0].transpose()[1]
        #
        # norm_x = [((x - min(X)) / (max(X) - min(X))) for x in X]
        # norm_y = [((y - min(Y)) / (max(Y) - min(Y))) for y in Y]

        # return np.dstack((norm_x, norm_y))

        return np.array(data)


class Distribution:

    def __init__(self, n, type="random"):
        self.n = 2 ** n
        self.distribution_dict = {}

        if type == "random":
            self.distribution = self.generate_random()
        elif type == "normal":
            self.distribution = self.generate_normal()
        elif type == "single_valued":
            self.distribution = self.generate_single_valued()

        for i in range(len(self.distribution)):
            self.distribution_dict[str(i)] = self.distribution[i]

    def generate_single_valued(self):
        distribution = np.zeros(self.n)
        distribution[randrange(0, self.n - 1)] = 1

        return distribution

    def generate_normal(self):
        tot_samples = 2000
        mu, sigma = self.n / 2, self.n / 8
        samples = np.random.normal(mu, sigma, tot_samples)

        distribution = np.zeros(self.n)

        for x in samples:
            if x > self.n - 1:
                x = self.n - 1
            if x < 0:
                x = 0
            distribution[int(x)] += 1 / tot_samples

        return distribution

    def generate_random(self):
        probs = np.random.random_sample((self.n,)) ** 2
        norm_fact = np.sum(probs)

        norm_probs = probs/norm_fact

        return norm_probs

    def plot_distribution(self):
        plt.bar(np.arange(len(self.distribution)), self.distribution)
        plt.show()


if __name__ == '__main__':
    dist = Distribution(3, type="normal")
    dist.plot_distribution()
    print(dist.distribution)


# data = TwoClusters().generate(100).transpose()
#
# plt.scatter(data[0], data[1])
# plt.xlim([0, 1])
# plt.ylim([0, 1])
# plt.show()
