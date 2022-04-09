import cv2
import numpy as np
from collections import deque

#define basic colors
red = (0, 0, 255)
green = (0, 255, 0)
blue = (255, 0, 0)
white = (255, 255, 255)
black = (0, 0, 0)

img = cv2.imread('maze.png',1)
img = cv2.resize(img,(50,50))
n,m,l = img.shape

def find_dist(point, current):
    return (point[0] - current[0])**2 + (point[1] - current[1])**2
    #Since we dont need the exact distance, we will keep it as square itself

def inRange(img,point):
    #checks if point is within the range of the image's dimensions
    if point[0]>=0 and point[0]<img.shape[0] and point[1]>=0 and point[1]<img.shape[1]:
        return True
    else:
        return False

def bfs(img,start,end):
    pass
def dfs(img,start,end):
    pass

def dj(img, start, end):
    #Note that in djikstra algorithm in image, the cost corresponds to distance between two pixels
    global n,m,l
    dist = np.full((n,m),fill_value = np.inf) #Stores distances of each pixel from start
    dist[start] = 0 
    parent = np.zeros((n,m,2)) #Stores x and y coordinates of parent, hence it has 3 dimensions
    visited = np.zeros((n,m)) #Stores whether a pixel has been visited or not
    visited[start] = 1
    current = start
    while current!=end:
        visited[current]=1 
        for i in range(-1,2): #-1,-0,1
            for j in range(-1,2): #-1,-0,1
                point = (current[0]+i,current[1]+j)
                if inRange(img,point) and visited[point]==0 and not(img[point][1]==white[1] and img[point][2]==white[2]):
                    if dist[point]>dist[current]+find_dist(point,current):
                        dist[point] = dist[current]+find_dist(point,current)
                        parent[point[0],point[1],0] = current[0]
                        parent[point[0],point[1],1] = current[1]

        #Finding the node with minimum distance, which will become the next node from which we check
        min = np.inf
        for i in range(n):
            for j in range(m):
                if dist[i,j] < min and visited[i,j] == 0:
                    min = dist[i,j]
                    current = (i,j)
        
    #now we will get the shortest path from the parent list
    curr_node = end
    shortest_path = []
    while curr_node!=start:
        shortest_path.append(curr_node)
        pair = int(parent[curr_node[0],curr_node[1],0]),int(parent[curr_node[0],curr_node[1],1])
        curr_node = (pair[0],pair[1])

    shortest_path.append(start)
    shortest_path.reverse()
    return shortest_path

cv2.namedWindow('image',cv2.WINDOW_NORMAL)
cv2.imshow('image',img.astype(np.uint8))

#get starting and ending positions: red = start, blue = end
flag1 = flag2 = True
for i in range(n):
    for j in range(m):
        if flag1:
            if(img[i,j]==red).all():
                st = (i,j)
                flag1 = False
        if flag2:
            if(img[i,j]==blue).all():
                en = (i,j)
                flag2 = False

img1=img
shortest_path = dj(img1, st, en)
for coordinate in shortest_path:
    img1[coordinate] = green

cv2.namedWindow('path',cv2.WINDOW_NORMAL)
cv2.imshow('path',img1.astype(np.uint8))

cv2.waitKey(0)
cv2.destroyAllWindows()