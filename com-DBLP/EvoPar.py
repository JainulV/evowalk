from LoclGraph import *
from SetWithBoundary import *
from GenerateSample import *
import numpy as np
from multiprocessing.pool import ThreadPool


def evopar(graph, v, k=100, phi=0.5, eps=0.95):
    T = int(eps * np.log(k / (6 * phi)))
    c = 1
    k_eps = 2 * k ** (1+eps/2) / c

    pool = ThreadPool()
    threads = []
    num_trials = int(k ** (eps / 2))
    for i in range(num_trials):
        # res = generate_sample(graph, v, T, np.inf, phi_eps, k_eps)
        threads.append(pool.apply_async(generate_sample, args=(graph, v, T, np.inf, phi, k_eps)))

    for j in range(len(threads)):
        res = threads[j].get()
        if res.volume() <= k_eps and res.conductance() <= phi:
            pool.terminate()
            return res


# G = LoclGraph('com-dblp.ungraph.txt')
# res_g = evopar(G, 2069, phi=0.7)
# print res_g.volume(), res_g.conductance()
# print res_g.membership
# print G.adjacency_list[2069]
