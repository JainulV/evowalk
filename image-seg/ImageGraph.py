import numpy as np
import matplotlib.pyplot as plt
import copy

from sklearn.feature_extraction import image
from sklearn.cluster import SpectralClustering
from scipy import misc
from scipy.sparse import coo_matrix


class ImageGraph(object):

    def __init__(self, img=None, dic=None):
        # img = misc.imread(image_file, flatten=True)
        # print img.shape


        if dic != None:
            self.graph = copy.deepcopy(dic)
            self.volume = 0
            for i in self.graph.values():
                self.volume += np.sum(map(lambda a: a[1], i))
        else:
            # We use a mask that limits to the foreground: the problem that we are
            # interested in here is not separating the objects from the background,
            # but separating them one from the other.
            mask = img.astype(bool)

            img = img.astype(float)
            img += 1 + 0.2 * np.random.randn(*img.shape)

            # Convert the image into a graph with the value of the gradient on the
            # edges.
            self.graph = image.img_to_graph(img, mask=mask)

            # Take a decreasing function of the gradient: we take it weakly
            # dependent from the gradient the segmentation is close to a voronoi
            self.graph.data = np.exp(-self.graph.data / self.graph.data.std())

            self.orig = copy.deepcopy(self.graph)
            # plt.matshow(self.graph.toarray())
            # plt.show()

            # clustering = SpectralClustering().fit(img.reshape(img.shape[0] * img.shape[1], 1))
            # X = clustering.affinity_matrix_
            # print X, X.shape
            # # X[X < 0.00001] = 0
            # self.graph = coo_matrix(X)
            # # plt.matshow(X)
            # # plt.show()

            # remove self loops
            self.graph.setdiag(np.zeros(self.graph.shape[0]))

            self.graph = self.graph.todok()

            kast = dict()
            self.volume = 0
            for i in self.graph.items():
                if i[0][0] in kast:
                    kast[i[0][0]].append(i)
                else:
                    kast[i[0][0]] = [i]
                self.volume += i[1]
            self.graph = kast

    def neighbors(self, v):
        return self.graph[v]

    def degree(self, v):
        return np.sum(map(lambda x: x[1], self.neighbors(v)))

    # def volume(self):
    #     return np.sum(map(lambda x: x[1], self.graph.values()))
