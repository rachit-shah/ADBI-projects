import sys
import time
import networkx as nx
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql import functions
from pyspark.sql.functions import *
from graphframes import *
from copy import deepcopy

sc=SparkContext("local", "degree.py")
sqlContext = SQLContext(sc)

def articulations(g, usegraphframe=False):
	# Get the starting count of connected components
	components = g.connectedComponents().groupBy("component").count()
	initialConnectedComponents = components.count()

	# Default version sparkifies the connected components process 
	# and serializes node iteration.
	if usegraphframe:
		# Get vertex list for serial iteration
		vertexList = g.vertices.map(lambda x: x.id).collect()

		# For each vertex, generate a new graphframe missing that vertex
		# and calculate connected component count. Then append count to
		# the output
		articulationList = []
		for x in vertexList:
			newVertices = g.vertices.filter(col("id") != lit(x))
			newEdges = g.edges.filter(~(col("src")==lit(x)) & ~(col("dst") == lit(x)))
			g2 = GraphFrame(newVertices,newEdges)
			newConnectedComponents = g2.connectedComponents().groupBy("component").count().count()
			articulationList.append((x,1) if newConnectedComponents > initialConnectedComponents else (x,0))
		
		return sqlContext.createDataFrame(articulationList,['id','articulation'])
	# Non-default version sparkifies node iteration and uses networkx 
	# for connected components count.
	else:
		nxG = nx.Graph()
		nxG.add_nodes_from(g.vertices.map(lambda x: x.id).collect())
		nxG.add_edges_from(g.edges.map(lambda x: (x.src,x.dst)).collect())
		#print("edges",g.edges.count(),"=",nxG.number_of_edges())
		#print("vertices",g.vertices.count(),"=",nxG.number_of_nodes())
		
		articulationList = []
		nodeList = list(nxG.nodes)
		for i in range(len(nodeList)):
			subList = nodeList[:i] + nodeList[i+1:]
			subGraph = nxG.subgraph(subList)
			newConnectedComponents = nx.number_connected_components(subGraph)
			articulationList.append((nodeList[i],1) if newConnectedComponents > initialConnectedComponents else (nodeList[i],0))
		return sqlContext.createDataFrame(articulationList,['id','articulation'])
			
				
filename = sys.argv[1]
lines = sc.textFile(filename)

pairs = lines.map(lambda s: s.split(","))
e = sqlContext.createDataFrame(pairs,['src','dst'])
e = e.unionAll(e.selectExpr('src as dst','dst as src')).distinct() # Ensure undirectedness 	

# Extract all endpoints from input file and make a single column frame.
v = e.selectExpr('src as id').unionAll(e.selectExpr('dst as id')).distinct()	

# Create graphframe from the vertices and edges.
g = GraphFrame(v,e)

#Runtime approximately 5 minutes
print("---------------------------")
print("Processing graph using Spark iteration over nodes and serial (networkx) connectedness calculations")
init = time.time()
df = articulations(g, False)
print("Execution time: %s seconds" % (time.time() - init))
print("Articulation points:")
articulationsDF = df.filter('articulation = 1')
articulationsDF.show(truncate=False)
print("Writing output to file articulations_out.csv")
articulationsDF.toPandas().to_csv("articulations_out.csv")
print("---------------------------")

#Runtime for below is more than 2 hours
print("Processing graph using serial iteration over nodes and GraphFrame connectedness calculations")
init = time.time()
df = articulations(g, True)
print("Execution time: %s seconds" % (time.time() - init))
print("Articulation points:")
df.filter('articulation = 1').show(truncate=False)
