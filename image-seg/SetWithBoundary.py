from ImageGraph import *
import numpy as np


class SetWithBoundary(object):

    def __init__(self, g):
        self.local_graph = g
        self.membership = set()
        self.boundary = dict()

    def add_vertex(self, v):
        self.membership.add(v)
        neighbors = self.local_graph.neighbors(v)
        for u in neighbors:
            strip_u = u[0][1]
            if self.is_member(strip_u):
                continue
            if strip_u in self.boundary:
                self.boundary[strip_u] += u[1]
            else:
                self.boundary[strip_u] = u[1]
        if v in self.boundary:
            del self.boundary[v]

    def remove_vertex(self, v):
        self.membership.remove(v)
        neighbors = self.local_graph.neighbors(v)
        for u in neighbors:
            strip_u = u[0][1]
            if u in self.boundary:
                self.boundary[strip_u] -= u[1]
            if np.less_equal(self.boundary[strip_u], 0):
                del self.boundary[strip_u]

    def symmetric_difference(self, D):
        for v in D:
            if self.is_member(v):
                self.remove_vertex(v)
            else:
                self.add_vertex(v)

    def union(self, D):
        for v in D:
            if not self.is_member(v):
                self.add_vertex(v)

    def boundary_vertex_list(self):
        return self.boundary.keys()

    def incident_edge_count(self, v):
        res = 0
        if v in self.boundary:
            res = self.boundary[v]
        elif v not in self.boundary and self.is_member(v):
            res = self.local_graph.degree(v)
        return res

    def is_member(self, v):
        return v in self.membership

    def transition_prob(self, v):
        return 0.5 * (self.incident_edge_count(v) / np.float(self.local_graph.degree(v))
                      + self.is_member(v))

    def volume(self):
        return np.sum(map(self.local_graph.degree, self.membership))

    def out(self):
        result = 0
        for v in self.membership:
            neighbors = np.array(map(lambda a: a[0][1], self.local_graph.neighbors(v)))
            outcident_edges = len(neighbors) - len(neighbors[map(self.is_member, neighbors)])
            result += outcident_edges
        return result

    def conductance(self):
        return self.out() / np.float(self.volume())
