from cmath import inf

def dijkstar(graph, start, end):
    value = {}          # value[i] =current distance from start to i
    parent ={}          # parent[i] = previous node in the path from start to i
    min_dist = {}      # min_value[i] = minimum distance from start to i
    visited = {}        # visited[i] = true if i is visited
    shortest_path = []
    infinity = inf      # infinity = inf
    for node in graph:
        value[node]=infinity
        parent[node]= start
        visited[node]=False
    value[start] = 0
   
    while(all(visited.values())==False):
        min_value_node = (min(value, key=value.get))
        visited[min_value_node] = True
        for adj_node in graph[min_value_node] :
            if not visited[adj_node]:
                if value[adj_node] > value[min_value_node] + graph[min_value_node][adj_node]:
                    value[adj_node] = value[min_value_node] + graph[min_value_node][adj_node]
                    parent[adj_node] = min_value_node
        min_dist[min_value_node] = value[min_value_node]
        value.pop(min_value_node)
        
        
    
    print("min_dist = ",min_dist)
    print("parent = ",parent)
    current_node = end
    while(current_node != start):
        shortest_path.append(current_node)
        current_node = parent[current_node]
    shortest_path.append(start)
    shortest_path.reverse()
    return(shortest_path)
        
    
graph = {
    'A': {'B': 1, 'C': 4},
    'B': {'C': 4, 'D': 2, 'E': 7},
    'C': {'B': 4, 'D': 3, 'E': 5},
    'D': {'B': 2, 'C': 3, 'E': 4, 'F': 6},
    'E': {'B': 7, 'D': 4, 'C': 5, 'F': 7},
    'F': {'E': 7, 'D': 6},
}

shortest_path = dijkstar(graph, 'A', 'F')
print("shortest_path is: ")
for i in range(len(shortest_path)):
    if(i!=len(shortest_path)-1):
     print(shortest_path[i],"-> ",end="")
    else:
        print(shortest_path[i])
print("\n")