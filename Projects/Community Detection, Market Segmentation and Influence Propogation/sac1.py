'''
Author: Rachit Shah
Email: rshah25@ncsu.edu
Project: Attributed Graph Community Detection / Market Segmentation
Date: 3/10/19
'''
#Import Libraries
from igraph import Graph, VertexClustering
import numpy as np
import pandas as pd 
import math
import sys

#Function to calculate cosine similarity of 2 lists  cosine = v1*v2 / ||v1||*||v2||
def cosine_similarity(v1,v2):
    list1 = list(v1.attributes().values())
    list2 = list(v2.attributes().values())
    sum1 = 0
    sum2 = 0
    sum3 = 0
    for i in range(len(list1)):
        sum1 += list1[i]**2
        sum2 += list2[i]**2
        sum3 += list1[i]*list2[i]
    return sum3/np.sqrt(sum1*sum2)

#Function to precalculate similarity matrix in advance
def compute_similarity_matrix(graph):
    #Initialize empty similarity matrix of size V*V
    sim_matrix = [[None for _ in range(len(graph.vs))] for _ in range(len(graph.vs))]

    #Calculate similarity using cosine. Similarity is commutative Sim(i,j)=Sim(j,i)
    for i in range(len(graph.vs)):
        for j in range(len(graph.vs)):
            if sim_matrix[i][j] is None:
                sim_matrix[i][j] = cosine_similarity(graph.vs[i],graph.vs[j])
                sim_matrix[j][i] = sim_matrix[i][j]
    return sim_matrix

#Function to calculate DeltaQNewman. Its the difference between old modularity and new modularity after moving a node to a cluster
def dq_newman(graph,community_list,alpha,vertex,community):
    old_modularity = graph.modularity(community_list, weights='weight')
    temp = community_list[vertex]
    community_list[vertex] = community
    new_modularity = graph.modularity(community_list, weights='weight')
    community_list[vertex] = temp
    return new_modularity - old_modularity

#Function to calculate DeltaQAttr. It is the sum of similarity Sum(Sim(i,x)) for all i belongs to given cluster/community 
def dq_attr(graph,community_list,alpha,vertex,community,sim_matrix):
    sum=0
    count=0
    for i in range(len(community_list)):
        if community_list[i] == community:
            sum += sim_matrix[vertex][i]
            count+=1
    return sum /(count*len(set(community_list)))   #Normalize by length of community


#Function to implement phase 1 of SAC1 algorithm
def phase1(graph,alpha,sim_matrix):
    #Add each node to their own communities
    community_list = [i for i in range(len(graph.vs))]
    #Stopping criteria - run till converges or 15 iterations
    converge = False
    i=0
    while not converge and i<15:
        i+=1
        #For each vertex in graph, consider which other community it can be added to
        for vertex in range(len(graph.vs)):
            com_v = community_list[vertex] #community of vertex
            #Find community which leads to max modularity gain
            max_dq = float(-math.inf)      
            max_com = None
            for community in community_list:
                #Consider only the communities other than the one vertex belongs to
                if com_v != community:
                    #Find modularity gain
                    dq = alpha * dq_newman(graph,community_list,alpha,vertex,community) + (1-alpha) * dq_attr(graph,community_list,alpha,vertex,community,sim_matrix)
                    #Set max
                    if max_dq < dq:
                        max_dq = dq
                        max_com = community
            #If positive maximum gain, add the vertex to the max community
            if max_dq > 0 and max_com:
                converge = False
                community_list[vertex] = max_com 
            #If negative gain, converge
            elif max_dq <= 0:
                converge = True

    return community_list

#Function to rebase clusters to start from 0 index since igraph needs the vertices' indices to be in range 0 to length
#E.g. [25,12,12,99,99] => [0,1,1,2,2]. I also store the mapping in mapped_clusters variable so we can reproduce the original 
#vertices in the cluster
def rebase_clusters(list):
    #Dictionary to store mappings
    newmapping = {}
    #New community list after mapping is done
    new_community_list = []
    #start index from 0
    count = 0
    #store mapping to original vertices (before phase 1)
    mapped_clusters = {}
    for i in range(len(list)):
        vertex = list[i]
        if vertex in newmapping:
            new_community_list.append(newmapping[vertex])
            mapped_clusters[newmapping[vertex]].append(i)
        else:
            newmapping[vertex] = count
            new_community_list.append(count)
            mapped_clusters[count] = [i]
            count+=1
    return new_community_list, mapped_clusters

#Function to implement Phase 2 of the SAC1 algorithm. In this phase, we join all vertices we grouped together in phase 1 
#in a cluster and replace all of them with one vertex using contract_vertices. The mean of attributes is taken for the resultant vertex
#We also recalculate the similarity matrix for the new nodes and perform phase 1 again
def phase2(graph,community_list,alpha,sim_matrix):
    #Rebase clusters to start from 0 index
    community_list, mapped_clusters = rebase_clusters(community_list)

    #Group all vertices in a cluster to one node and simplify graph
    graph.contract_vertices(community_list,combine_attrs="mean")
    graph.simplify(combine_edges=sum,multiple=True,loops=True)

    #Recalculate similarity matrix for new vertices
    sim_matrix = compute_similarity_matrix(graph)

    #Add each node to their own communities
    community_list = [i for i in range(len(graph.vs))]

    #Peform phase 1 again
    community_list = phase1(graph,alpha,sim_matrix)
    return community_list, mapped_clusters
        
#Main Function
def main():
    #Check input parameter alpha
    if len(sys.argv)!=2:
        print("Enter value of alpha as parameter")
        return
    alpha = float(sys.argv[1])
    #Convert for preferred output to file
    alph = 0
    if alpha == 0.5:
        alph = 5
    elif alpha == 0.0:
        alph = 0
    elif alpha == 1.0:
        alph = 1
    
    #Input Graph
    graph = Graph.Read_Edgelist('data/fb_caltech_small_edgelist.txt')
    attr = pd.read_csv('data/fb_caltech_small_attrlist.csv')

    #Initialize weights and attributes
    graph.es['weight'] = 1
    attr_names = attr.keys()
    for x in attr.iterrows():
        for y in range(len(x[1])):
            graph.vs[x[0]][attr_names[y]] = x[1][y]

    #Similarity Matrix
    sim_matrix = compute_similarity_matrix(graph)

    #Phase 1
    community_list = phase1(graph,alpha,sim_matrix)
    print('Communities after Phase 1:')
    print(len(set(community_list)),"Communities")
    phase1_output = ''
    for x in VertexClustering(graph,community_list):
        if x:
            phase1_output += ','.join(map(str,x))
            phase1_output += "\n"
    phase1_output = phase1_output[:-2]

    #Phase 2
    community_list, mapped_clusters = phase2(graph,community_list,alpha,sim_matrix)
    print(mapped_clusters)
    print('Communities after Phase 2:')
    print(len(set(community_list)),"Communities")
    phase2_output = ''
    for cluster in VertexClustering(graph,community_list):
        if cluster:
            original_vertices = []
            for vertex in cluster:
                original_vertices.extend(mapped_clusters[vertex])
            phase2_output += ','.join(map(str,original_vertices))
            phase2_output += '\n'
            print(cluster)
    phase2_output = phase2_output[:-2]

    file = open("communities_"+str(alph)+".txt", 'w+')
    file.write(phase2_output)
    file.close()
    return
    

if __name__ == "__main__":
	main()