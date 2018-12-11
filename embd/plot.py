from LoclGraph import *
import numpy as np
from gensim.models import Word2Vec, KeyedVectors
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

model = KeyedVectors.load_word2vec_format('karate.emd', binary=False)
graph = LoclGraph()
graph.load_adjfile('karate.adjlist')
features_matrix = np.asarray([model[str(node)] for node in graph.vertex_set])

color = ['c', 'c', 'c', 'c', 'b', 'b', 'b', 'c', 'm', 'g',
         'b', 'c', 'c', 'c', 'm', 'm', 'b', 'c', 'm', 'c',
         'm', 'c', 'm', 'm', 'g', 'g', 'm', 'g', 'g', 'm',
         'm', 'g', 'm', 'm']
plt.scatter(features_matrix[:, 0], features_matrix[:, 1], color=color)
for i in np.arange(len(features_matrix[:, 0])):
    plt.annotate(i + 1, (features_matrix[i, 0], features_matrix[i, 1]))
plt.show()

G = nx.Graph()
G.add_edges_from(graph.edge_list)
nx.draw(G, pos=nx.spring_layout(G), with_labels=True, node_color=color)
plt.show()
