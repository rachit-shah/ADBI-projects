from igraph import *

g = Graph()
g.add_vertices(3)
g.add_edges([(0,1),(1,2)])
test_nodes = g.vs.indices
print("before",(test_nodes))
test_nodes[0]= test_nodes[1]
print("after",test_nodes)

test_cluster = VertexClustering(g,test_nodes)

print("Clusters",[c for c in test_cluster])

g.contract_vertices(test_nodes)
print("after",g.vs.indices)

g.simplify(combine_edges=sum)
print("after",g.vs.indices)
