#make a circle of radius 400 in the centre of an 800x800 screen 
import cv2
import numpy as np
import math
matrix = np.full((800,800),255)


for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
    # we will use eqn of circle to check if pixel needs to be colored (white)
    #(400,400) is centre, 400 is radius
        if(math.sqrt((400-i)**2+(400-j)**2)<=200):
            matrix[i,j] = 0

cv2.namedWindow('Circle',cv2.WINDOW_NORMAL)
cv2.imshow('Circle',matrix.astype(np.uint8))

cv2.waitKey(0)
cv2.destroyAllWindows()