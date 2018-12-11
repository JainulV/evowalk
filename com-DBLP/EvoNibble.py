from LoclGraph import *
from SetWithBoundary import *
from GenerateSample import *
import numpy as np


def evonibble(graph, vertx, phi=0.5, eps=0.5):
    T = int(eps * np.log(graph.volume / (3 * phi)))
    tot = graph.volume
    ver = list()
    prob = list()
    for vert, p in vertx.items():
        ver.append(vert)
        prob.append(np.float(p))
    prob = np.array(prob) / np.sum(prob)
    X = np.random.choice(ver, p=prob)
    J = np.random.uniform(0, np.log2(tot))
    alpha = 1 + 4 * np.sqrt(T * np.log(tot))
    beta = 2 * tot ** eps
    B = 16 * alpha * beta * (2 ** J)
    S = generate_sample(graph, X, T, B, phi, np.inf)
    if S.conductance() <= phi and S.volume() <= tot / 2.:
        return S
    else:
        return SetWithBoundary(graph)


def evopartition(graph, phi=0.5, eps=1.e-2):
    V = {k: graph.degree(k) for k in graph.adjacency_list.keys()}
    S = SetWithBoundary(graph)
    l = int(graph.volume ** (1+2*eps))
    for j in range(l):
        P = evonibble(graph, V, phi)
        S.union(P.membership)
        if S.membership:
            print S.conductance(), S.volume()
        for vert in P.membership:
            if vert in V:
                del V[vert]
        if S.volume() >= graph.volume / 2:
            return S
    return S


# G = LoclGraph('com-dblp.ungraph.txt')
# print G.volume
# res = evopartition(G, phi=0.01)
# if res:
#     print res.conductance(), res.volume()
#     print res.membership

G = LoclGraph('com-dblp.ungraph.txt')
print G.volume
eps = 0.01
V = {k: G.degree(k) for k in G.adjacency_list.keys()}
l = int(G.volume ** (1+2*eps))
for j in range(l):
    print j
    res = evonibble(G, V, 0.01)
    if res.membership:
        print res.conductance(), res.volume()
        print res.membership
        break
