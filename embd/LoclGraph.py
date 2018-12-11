import numpy as np
from scipy.io import loadmat


class LoclGraph(object):
    def __init__(self):
        self.vertex_set, self.edge_list = set(), list()
        self.adjacency_list = dict()
        self.volume = 0

    def load_adjfile(self, file_):
        with open(file_) as f:
            for line in f:
                if line[0] != '#':
                    contents = map(int, line.strip().split(' '))
                    self.adjacency_list[contents[0]] = contents[1:]
                    self.vertex_set.add(contents[0])
        self.edge_list = [(i, j) for i in self.adjacency_list for j in self.adjacency_list[i]]
        self.volume = np.sum(map(len, self.adjacency_list.values()))

    def load_edgefile(self, file_):
        with open(file_) as f:
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

    def load_matfile(self, file_, variable_name='network'):
        mat_variables = loadmat(file_)
        mat_matrix = mat_variables[variable_name]

        cx = mat_matrix.tocoo()
        for i, j, v in zip(cx.row, cx.col, cx.data):
            self.vertex_set.update((i, j))
            self.edge_list.append((i, j))
        for vertex in self.vertex_set:
            self.adjacency_list[vertex] = list()
        for i, j, v in zip(cx.row, cx.col, cx.data):
            self.adjacency_list[i].append(j)
            self.adjacency_list[j].append(i)

        self.volume = np.sum(map(len, self.adjacency_list.values()))

    def neighbors(self, v):
        return self.adjacency_list[v]

    def degree(self, v):
        return len(self.neighbors(v))
