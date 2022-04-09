import cv2 
import numpy as np
from collections import deque

img = cv2.imread('test.png',1)
img1 = img
class node():
    def __init__(self, index, parent):
        self.x = index[0]
        self.y = index[1]
        self.parent = parent
    

def show_path(start, end):
    print(f"Start: ({start.x},{start.y})")
    print(f"End: ({end.x},{end.y})")
    current = start
    while current!=start:
        img1[current.x][current.y] = [0,255,0] #Setting to green

def bfs(start):
    q = deque()
    pass

def dfs(start):
    pass

