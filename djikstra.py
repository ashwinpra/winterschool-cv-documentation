import cv2
import numpy as np
from cmath import inf
graph = {
    'A': {'B': 1, 'C': 4},
    'B': {'C': 4, 'D': 2, 'E': 7},
    'C': {'B': 4, 'D': 3, 'E': 5},
    'D': {'B': 2, 'C': 3, 'E': 4, 'F': 6},
    'E': {'B': 7, 'D': 4, 'C': 5, 'F': 7},
    'F': {'E': 7, 'D': 6},
}

def dj(graph, start, end):
    value = {} #stores current distance from start to each ndoe (not necessarily shortest)
    min_dist = {} #stores minimum distance from start to each node
    parent = {} #stores node that was just before current node
    visited = {} #stores "1" for every visited node, otherwise "0"
    shortest_path = [] #stores shortest path to reach each node
    infty = inf 

    #initialising all values to infty except start (=0)
    for node in graph:
        if node == start:
            value[node] = 0
        value[node] = infty
        parent[node] = start
        visited[node] = False
    
    #iterating till each node is visited
    while all(visited.values()) == False:
        #find node with minimum distance
        min_dist = inf
        for node in graph:
            if visited[node] == 0 and value[node] < min_dist:
                min_dist = value[node]
                min_dist_node = node
        #mark this node as visited
        visited[min_dist_node] = False
        #update distance to each neighbour if its lower than current dist
        for neighbour in graph[min_dist_node]:
            if visited[neighbour] == 0:
                # if value(min dist node) + cost < value (neighbour) then update
                if value[min_dist_node] + graph[min_dist_node][neighbour] < value[neighbour]:
                    value[neighbour] = value[min_dist_node] + graph[min_dist_node][neighbour]
                    # also update its parent to this node if its lesser
                    parent[neighbour] = min_dist_node
        
    
    # We are now done with the traversal, so we will print/return all the results
    print("Minimum distance of each ndoe from start: ")
    print(min_dist)
    print("Parent: ")
    print(parent)

    curr_node = end
    while curr_node != start:
        shortest_path.append(curr_node)
        curr_node = parent[curr_node]
    
    print("Shortest path from start to end: ")
    print(shortest_path)

    return
    
dj(graph,'A','F')