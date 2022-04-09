from cmath import inf

def dj(graph, start, end):
    value = {} #stores current distance from start to each node (not necessarily shortest)
    min_dist = {} #stores minimum distance from start to each node
    parent = {} #stores node that was just before current node
    visited = {} #stores "1" for every visited node, otherwise "0"
    shortest_path = [] #stores shortest path to reach each node
    infty = inf 

    #initialising all values to infty except start (=0)
    for node in graph:
        value[node] = infty
        parent[node] = start
        visited[node] = False
    value[start] = 0

    #iterating till each node is visited
    while (all(visited.values()) == False):
        #find node with minimum distance
        min_dist_node  = (min(value, key=value.get))
        #mark this node as visited
        visited[min_dist_node] = True
        #update distance to each neighbour if its lower than current dist
        for neighbour in graph[min_dist_node]:
            if not visited[neighbour]:
                # if value(min dist node) + cost < value (neighbour) then update
                if value[min_dist_node] + graph[min_dist_node][neighbour] < value[neighbour]:
                    value[neighbour] = value[min_dist_node] + graph[min_dist_node][neighbour]
                    # also update its parent to this node if its lesser
                    parent[neighbour] = min_dist_node
        min_dist[min_dist_node] = value[min_dist_node]
        value.pop(min_dist_node)
    
    # We are now done with the traversal, so we will print/return all the results
    print("Minimum distance of each node = ",min_dist)
    print("parent = ",parent)

    curr_node = end
    while (curr_node != start):
        shortest_path.append(curr_node)
        curr_node = parent[curr_node]
    shortest_path.append(start)
    shortest_path.reverse()

    return shortest_path


graph = {
    'A': {'B': 1, 'C': 4},
    'B': {'C': 4, 'D': 2, 'E': 7},
    'C': {'B': 4, 'D': 3, 'E': 5},
    'D': {'B': 2, 'C': 3, 'E': 4, 'F': 6},
    'E': {'B': 7, 'D': 4, 'C': 5, 'F': 7},
    'F': {'E': 7, 'D': 6},
}

   
shortest_path = dj(graph, 'A', 'F')
print("shortest path =  ",shortest_path)