from LoclGraph import *
import numpy as np
from gensim.models import Word2Vec, KeyedVectors
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from scipy.io import loadmat
from sklearn.preprocessing import MultiLabelBinarizer
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


emd_file = 'blogcatalog.emd'
mat_file = 'blogcatalog.mat'
train_percents = np.arange(0.1, 1, step=0.1)

# load the embeddings and the corresponding graph
model = KeyedVectors.load_word2vec_format(emd_file, binary=False)
graph = LoclGraph()
graph.load_matfile(mat_file)

# load the labels
mat = loadmat(mat_file)
labels_matrix = mat['group']
labels_count = labels_matrix.shape[1]
print labels_matrix.toarray()
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
    top_k_list = [l.getnnz() for l in y_test]
    preds = clf.predict(X_test, top_k_list)

    results = {}
    averages = ['micro', 'macro']
    mlb = MultiLabelBinarizer(range(labels_count))
    for average in averages:
        results[average] = f1_score(y_test, mlb.fit_transform(preds), average=average)
    all_results[train_percent] = results

print all_results
