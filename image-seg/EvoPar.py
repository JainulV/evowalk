from ImageGraph import *
from SetWithBoundary import *
from GenerateSample import *
import numpy as np
from multiprocessing import Pool


def evopar(graph, v, k=100, phi=0.5, eps=0.95):
    T = max(int(eps * np.log(k / (3 * phi))), 10)
    c = 1
    k_eps = 2 * k ** (1+eps/2) / c

    pool = Pool()
    threads = []
    num_trials = int(k ** (eps / 2))
    for i in range(num_trials):
        # res = generate_sample(graph, v, T, np.inf, phi_eps, k_eps)
        threads.append(pool.apply_async(generate_sample, args=(graph, v, 10, np.inf, phi, k_eps)))

    for j in range(len(threads)):
        res = threads[j].get()
        if res.volume() <= k_eps and res.conductance() <= phi:
            pool.terminate()
            return res


G = ImageGraph('wolf.jpg')
res_g = evopar(G, 1, phi=0.3)
print res_g.volume(), res_g.conductance()
print res_g.membership
