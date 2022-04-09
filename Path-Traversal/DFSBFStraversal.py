import cv2
import numpy as np
from DS import Queue
from DS import Stack

#define basic colors
red = (0, 0, 255)
green = (0, 255, 0)
blue = (255, 0, 0)
white =(255, 255, 255)
black = (0, 0, 0)

img = cv2.imread('maze.png',1)
img = cv2.resize(img,(25,25))
n,m,l = img.shape


def bfs(start):
    q = Queue()
    q.enqueue(start)
    visited = np.zeros((n,m))
    pass

def dfs(start):
    s = Stack()

    pass


cv2.namedWindow('image',cv2.WINDOW_NORMAL)
cv2.imshow('image',img)

#get starting and ending positions: red = start, blue = end
for i in range(n):
    for j in range(m):
        if(img[i,j]==red).all():
            st = (i,j)
        if(img[i,j]==blue).all():
            en = (i,j)


cv2.namedWindow('path',cv2.WINDOW_NORMAL)
cv2.imshow('path',img)

cv2.waitKey(0)
cv2.destroyAllWindows()