from ImageGraph import *
from SetWithBoundary import *
from GenerateSample import *
import numpy as np
from mnist import MNIST


def evonibble(graph, vertx, phi=0.5, eps=0.5):
    T = int(eps * np.log(graph.volume / (3 * phi)))
    tot = graph.volume
    ver = list()
    prob = list()
    for vert, p in vertx.items():
        ver.append(vert)
        prob.append(p)
    prob = np.array(prob) / np.sum(prob)
    X = np.random.choice(ver, p=prob)
    J = np.random.uniform(0, np.log2(tot))
    alpha = 1 + 4 * np.sqrt(T * np.log(tot))
    beta = 2 * tot ** eps
    B = 16 * alpha * beta * (2 ** J)
    S = generate_sample(graph, X, T, B, phi, np.inf)
    # print S.membership, S.conductance()
    if S.conductance() <= phi and S.volume() <= tot / 2.:
        return S
    else:
        return SetWithBoundary(graph)


def evopartition(graph, keys, phi=0.5, eps=1.e-2):
    V = {k: graph.degree(k) for k in keys}
    S = SetWithBoundary(graph)
    l = int(graph.volume ** (1+2*eps))
    for j in range(l):
        P = evonibble(graph, V, phi)
        S.union(P.membership)
        # if S.membership:
        #     print S.conductance(), S.volume()
        for vert in P.membership:
            if vert in V:
                del V[vert]
        if len(S.membership) >= len(keys) / 2:
            return S
    return S


def rec(graph, keys, iter, phi=0.5, eps=1.e-2):
    global rl
    R = evopartition(graph, keys, phi, eps)
    if R.membership and R.conductance() >= 0.9 or iter >= 2:
        rl.append(R.membership)
        return
    R_comp = np.setdiff1d(np.array(graph.graph.keys()), np.array(list(R.membership)))
    # R_dict, R_comp_dict = dict(), dict()
    # for k, v in graph.graph.items():
    #     if k in R.membership:
    #         R_dict[k] = v
    # for k, v in graph.graph.items():
    #     if k in R_comp:
    #         R_comp_dict[k] = v
    # R_graph = ImageGraph(dic=R_dict)
    # R_comp_graph = ImageGraph(dic=R_comp_dict)
    rec(graph, np.array(list(R.membership)), iter + 1, phi, eps)
    rec(graph, R_comp, iter + 1, phi, eps)


# #############################################################################
l = 100
x, y = np.indices((l, l))

center1 = (28, 24)
center2 = (40, 50)
center3 = (67, 58)
center4 = (24, 70)

radius1, radius2, radius3, radius4 = 16, 14, 15, 14

circle1 = (x - center1[0]) ** 2 + (y - center1[1]) ** 2 < radius1 ** 2
circle2 = (x - center2[0]) ** 2 + (y - center2[1]) ** 2 < radius2 ** 2
circle3 = (x - center3[0]) ** 2 + (y - center3[1]) ** 2 < radius3 ** 2
circle4 = (x - center4[0]) ** 2 + (y - center4[1]) ** 2 < radius4 ** 2

# #############################################################################
# 4 circles
img = circle1 + circle2 + circle3 + circle4


def load_dataset():
    mndata = MNIST('../data/')
    X_train, labels_train = map(np.array, mndata.load_training())
    X_test, labels_test = map(np.array, mndata.load_testing())
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    return X_train[0].reshape((28, 28))


# img = load_dataset()
mask = img.astype(bool)
G = ImageGraph(img)
print G.volume
res = evopartition(G, G.graph.keys())
if res:
    print res.conductance(), res.volume()
    print res.membership
    i = np.zeros((2678,))
    i[list(res.membership)] = 1
    label_im = np.full(mask.shape, -1.)
    label_im[mask] = i
    plt.matshow(label_im)
    plt.show()

rl = list()
rec(G, G.graph.keys(), 0)
i = np.zeros((2678,))    # 166
f = 1
for _ in rl:
    i[list(_)] = f
    f += 1
label_im = np.full(mask.shape, -1.)
label_im[mask] = i
plt.matshow(label_im)
plt.show()
