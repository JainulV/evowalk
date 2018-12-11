from ImageGraph import *
from SetWithBoundary import *
from GenerateSample import *

G = ImageGraph('./wolf.jpg')
print G.degree(0)
# print G.graph, G.graph.shape, G.volume(), G.neighbors(0)
S = SetWithBoundary(G)
S.add_vertex(0)
print S.membership, S.boundary
print S.transition_prob(481)
print '------'
S.add_vertex(481)
print S.conductance()
print S.membership, S.boundary

S = generate_sample(G, 2070)
print S.membership, S.conductance(), S.volume()
