from LoclGraph import *
from SetWithBoundary import *
from GenerateSample import *
import numpy as np
from multiprocessing import Pool
from EvoPar import *


G = LoclGraph('com-dblp.ungraph.txt')
communities = []
with open('com-dblp.all.cmty.txt') as f:
    for line in f:
        communities.append(map(int, line.strip().split('\t')))
itr = 0
print len(communities)
for community in communities:
    X = np.random.choice(community)
    for phi in np.arange(0.1, 1, step=0.1):
        R = evopar(G, X, phi=phi)
        if R: break
    # print X, R.membership, R.conductance(), community
    tpr = 1 - len(np.setdiff1d(np.array(community), np.array(list(R.membership)))) / np.float(len(community))
    fdr = len(np.setdiff1d(np.array(list(R.membership)), np.array(community))) / np.float(len(R.membership))
    print tpr, fdr
    itr += 1
    # if itr == 1: break
