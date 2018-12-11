from ImageGraph import *
from SetWithBoundary import *
import numpy as np


def generate_sample(G, v, T=10, B=500, phi=0.5, K=10):
    S = SetWithBoundary(G)
    S.add_vertex(v)
    X = v
    cost = S.volume()
    for t in range(T):
        # Stage 1: Select the vertices to add or remove from S
        rand = np.random.choice([0, 1])
        if rand:
            neigh = G.neighbors(X)
            tot = G.degree(X)
            nxt = map(lambda x: x[0][1], neigh)
            prob = map(lambda x: x[1] / tot, neigh)
            X = np.random.choice(nxt, p=prob)
        transit = S.transition_prob(X)
        Z = np.random.uniform(0, transit)
        St = SetWithBoundary(G)
        for u in S.membership:
            if S.transition_prob(u) >= Z:
                St.add_vertex(u)
        for u in S.boundary_vertex_list():
            if S.transition_prob(u) >= Z:
                St.add_vertex(u)
        D = list()
        symm = S.membership.symmetric_difference(St.membership)
        for u in S.boundary_vertex_list():
            if S.transition_prob(u) >= Z and u in symm:
                D.append(u)
        cost += St.volume() + S.out()
        if cost > B:
            S.symmetric_difference(D)
            break

        # Stage 2: Update S
        S.symmetric_difference(D)
        if S.conductance() <= phi and S.volume() <= K:
            return S
    return S
