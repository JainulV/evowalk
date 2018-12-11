from LoclGraph import *
from SetWithBoundary import *
from GenerateSample import *

G = LoclGraph('com-dblp.ungraph.txt')

D = generate_sample(G, 0)
print D.membership, D.conductance(), D.volume()
