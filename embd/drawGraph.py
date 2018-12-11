from LoclGraph import *
from SetWithBoundary import *
from GenerateSample import *
from EvoPar import *
import numpy as np
from gensim.models import Word2Vec
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool


G = LoclGraph()
# G.load_matfile('blogcatalog.mat')
# G.load_edgefile('p2p-Gnutella08.edgelist')
G.load_adjfile('karate.adjlist')


def build_deepwalk_corpus(G, num_paths):
    # def generate(G):
    #     nodes = list(G.vertex_set)
    #     np.random.shuffle(nodes)
    #     res = []
    #     for node in nodes:
    #         res.append(tuple(map(str, generate_sample(G, node).membership)))
    #     return res

    # walks = list()
    # pool = ThreadPool(1)
    # results = pool.map(lambda _: generate(G), range(num_paths))
    # pool.terminate()
    # pool.join()
    # for result in results:
    #     walks.extend(result)
    walks = list()
    nodes = list(G.vertex_set)
    for _ in range(num_paths):
        print _
        np.random.shuffle(nodes)
        res = []
        for node in nodes:
            res.append(tuple(map(str, generate_sample(G, node).membership)))
        walks.extend(res)
    return walks


print cpu_count()

print("Walking...")
walks = build_deepwalk_corpus(G, 150)
# print walks
print("Training...")
model = Word2Vec(walks, size=2, window=10, min_count=0, sg=1, hs=1, workers=cpu_count())
model.wv.save_word2vec_format('karate-2.emd')
