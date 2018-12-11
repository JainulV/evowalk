from LoclGraph import *
import numpy as np
from gensim.models import Word2Vec, KeyedVectors
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from scipy.io import loadmat
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.cluster import SpectralClustering
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


class TopKRanker(OneVsRestClassifier):
    # borrowed from https://github.com/phanein/deepwalk/blob/master/example_graphs/scoring.py
    # for comparison study
    def predict(self, X, top_k_list):
        assert X.shape[0] == len(top_k_list)
        probs = np.asarray(super(TopKRanker, self).predict_proba(X))
        all_labels = []
        for i, k in enumerate(top_k_list):
            probs_ = probs[i, :]
            labels = self.classes_[probs_.argsort()[-k:]].tolist()
            all_labels.append(labels)
        return all_labels


emd_file = 'karate-2.emd'
adj_file = 'karate.adjlist'
train_percents = [0.8]

# load the embeddings and the corresponding graph
model = KeyedVectors.load_word2vec_format(emd_file, binary=False)
graph = LoclGraph()
graph.load_adjfile(adj_file)

# load the labels
classes = ['c', 'c', 'c', 'c', 'b', 'b', 'b', 'c', 'm', 'g',
           'b', 'c', 'c', 'c', 'm', 'm', 'b', 'c', 'm', 'c',
           'm', 'c', 'm', 'm', 'g', 'g', 'm', 'g', 'g', 'm',
           'm', 'g', 'm', 'm']
m = {'c': 0, 'b': 1, 'g': 2, 'm': 3}
labels_matrix = np.array(map(lambda x: (m[x], ), classes))
mlb = MultiLabelBinarizer(range(len(m)))
labels_matrix = mlb.fit_transform(labels_matrix)
labels_count = labels_matrix.shape[1]
# Map nodes to their features
features_matrix = np.asarray([model[str(node)] for node in graph.vertex_set])


# perform learning and evaluation
all_results = dict()
perm = np.random.permutation(np.arange(len(graph.vertex_set)))
for train_percent in train_percents:
    # create test-train splits
    split_point = int(train_percent * len(graph.vertex_set))
    train_idx, test_idx = perm[:split_point], perm[split_point:]
    X_train, y_train = features_matrix[train_idx, :], labels_matrix[train_idx, :]
    X_test, y_test = features_matrix[test_idx, :], labels_matrix[test_idx, :]

    # train the model
    clf = TopKRanker(LogisticRegression())
    clf.fit(X_train, y_train)

    # find out how many labels should be predicted
    top_k_list = [np.count_nonzero(l) for l in y_test]
    preds = clf.predict(X_test, top_k_list)

    xx, yy = np.mgrid[-3:3:.01, -3:3:.01]
    grid = np.c_[xx.ravel(), yy.ravel()]
    g = [1 for _ in range(grid.shape[0])]
    probs = np.array(clf.predict(grid, g)).reshape(xx.shape)
    f, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(xx, yy, probs, cmap="RdBu",
                        vmin=0, vmax=1)
    ax_c = f.colorbar(contour)
    ax_c.set_label("$P(y = 1)$")
    ax_c.set_ticks([0, .25, .5, .75, 1])

    ax.scatter(X_train[:,0], X_train[:, 1], c=y_train, s=50,
            cmap="RdBu", vmin=-.2, vmax=1.2,
            edgecolor="white", linewidth=1)

    ax.set(aspect="equal",
        xlim=(-3, 3), ylim=(-3, 3),
        xlabel="$X_1$", ylabel="$X_2$")
    plt.show()

    results = {}
    averages = ['micro', 'macro']
    mlb = MultiLabelBinarizer(range(labels_count))
    for average in averages:
        results[average] = f1_score(y_test, mlb.fit_transform(preds), average=average)
    all_results[train_percent] = results

print all_results

# now the spectral clustering algorithm for comparison
clustering = SpectralClustering(n_clusters=4,
                                random_state=0).fit(features_matrix)
ll = map(lambda x: (x, ), clustering.labels_)
print clustering.labels_
