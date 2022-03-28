import cv2
import numpy as np

img = cv2.imread('test.png',1)
imgMatrix = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
n,m = imgMatrix.shape
imgX = np.full(img.shape,0)
imgY = np.full(img.shape,0)
imgXY = np.full(img.shape,0)

# Method 1: Manual 
for i in range(n):
    for j in range(m):
        imgX[i,j] = imgMatrix[n-i-1,j] # reversed about X
        imgY[i,j] = imgMatrix[i,m-j-1] # reversed about Y
        imgXY[i,j] = imgMatrix[n-i-1,m-j-1] # reversed about X and Y

cv2.namedWindow('About X', cv2.WINDOW_NORMAL)
cv2.imshow('About X', imgX.astype(np.uint8))

cv2.namedWindow('About Y', cv2.WINDOW_NORMAL)
cv2.imshow('About Y', imgY.astype(np.uint8))

cv2.namedWindow('About XY', cv2.WINDOW_NORMAL)
cv2.imshow('About XY', imgXY.astype(np.uint8))

cv2.waitKey(0)
cv2.destroyAllWindows()

# Method 2: using cv2.flip(src, flipCode)
# flipCode: 0=>Vertical 1=>Horizontal -1=>Both