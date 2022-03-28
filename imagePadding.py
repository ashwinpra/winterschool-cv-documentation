import cv2
import numpy as np

# Method 1: Create a larger matrix, and add specific color to the borders 

img = cv2.imread('test.png',1)
n,m,l = img.shape
n+=40
m+=40
matrix = np.full((n,m,l),0)
# 490 x 640 

for i in range(n):
     for j in range(m):
         if (i>20 and i<n-20) and (j>20 and j<m-20):
            matrix[i,j,0] = img[i-20,j-20,0]
            matrix[i,j,1] = img[i-20,j-20,1]
            matrix[i,j,2] = img[i-20,j-20,2]


cv2.namedWindow('Border 1',cv2.WINDOW_NORMAL)
cv2.imshow('Border 1',matrix.astype(np.uint8))

# Method 2: Use inbuilt functions
new = cv2.copyMakeBorder(img,20,20,20,20,cv2.BORDER_CONSTANT,None,(255,0,0))

# after img, 4 numbers are the distances of border from top,bottom,left,right
# Border types:
# BORDER_NORMAL => Constant colored border -> Color will be in (B,G,R) form
# BORDER_REFLECT => Border will be mirror reflection of the image
# BORDER_REFLECT_101 or BORDER_DEFAULT => Slightly different from BORDER_REFLECT
# BORDER_REPLICATE => Replicates the last element

cv2.namedWindow('Border 2',cv2.WINDOW_NORMAL)
cv2.imshow('Border 2',new.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()