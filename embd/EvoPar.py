from LoclGraph import *
from SetWithBoundary import *
from GenerateSample import *
import numpy as np
from multiprocessing.pool import ThreadPool
from itertools import compress
from multiprocessing import cpu_count


def evopar(graph, v, k=100, phi=0.5, eps=1.):
    T = int(eps * np.log(k / (6 * phi)))
    c = 1
    k_eps = 2 * k ** (1+eps/2) / c

    pool = ThreadPool(cpu_count())
    num_trials = int(k ** (eps / 2))

    results = pool.map(lambda _: generate_sample(graph, v, 1, np.inf, phi, k_eps), np.arange(num_trials))
    pool.close()
    pool.join()

    return min(results, key=lambda x: x.conductance())


# G = LoclGraph('com-dblp.ungraph.txt')
# res_g = evopar(G, 2069, phi=0.7)
# print res_g.volume(), res_g.conductance()
# print res_g.membership
# print G.adjacency_list[2069]
