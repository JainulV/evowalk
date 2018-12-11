import numpy as np


class LoclGraph(object):
    def __init__(self, filename):
        self.vertex_set, self.edge_list = set(), list()
        with open(filename) as f:
            for line in f:
                if line[0] != '#':
                    contents = map(int, line.strip().split('\t'))
                    self.vertex_set.update(contents)
                    self.edge_list.append(tuple(contents))
        self.adjacency_list = dict()
        for vertex in self.vertex_set:
            self.adjacency_list[vertex] = list()
        for edge in self.edge_list:
            self.adjacency_list[edge[0]].append(edge[1])
            self.adjacency_list[edge[1]].append(edge[0])

        self.volume = np.sum(map(len, self.adjacency_list.values()))

    def neighbors(self, v):
        return self.adjacency_list[v]

    def degree(self, v):
        return len(self.neighbors(v))
