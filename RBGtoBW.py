import cv2
import numpy as np

img = cv2.imread('rgbcube.png',1)
# image is actually in b,g,r form so b->0 and r->2
# also it is a 3d-3d array of which we need only the corresponding color arrays
r, g, b = img[:,:,2], img[:,:,1], img[:,:,0]
n,m,l = img.shape


# Method 1: average of R,G,B
matrix1 = np.full((n,m),255) #to avoid the third dimension of shape (which is no. of channels - RGB)
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        matrix1[i,j] = 0.33*r[i,j]+0.33*g[i,j]+0.33*b[i,j]
        #to avoid overflow (>255) we are dividing individually

cv2.namedWindow('Method 1',cv2.WINDOW_NORMAL)
cv2.imshow('Method 1',matrix1.astype(np.uint8))

# Method 2: Since human eye can perceive green color the most and blue the least, 
# Grey = 0.21R + 0.72G + 0.07B OR 
# Grey = 0.299R + 0.587G + 0.114B 

matrix2 = np.full((n,m),255) #to avoid the third dimension of shape (which is no. of channels - RGB)
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        matrix2[i,j] = 0.299*r[i,j]+0.587*g[i,j]+0.114*b[i,j]
        #to avoid overflow (>255) we are dividing individually

cv2.namedWindow('Method 2',cv2.WINDOW_NORMAL)
cv2.imshow('Method 2',matrix2.astype(np.uint8))  

# Method 3: Take mean of greatest and lowest values: max+min/2
matrix3 = np.full((n,m),255) #to avoid the third dimension of shape (which is no. of channels - RGB)
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        matrix3[i,j] = max(r[i,j],g[i,j],b[i,j])/2 + min(r[i,j],g[i,j],b[i,j])/2
        #to avoid overflow (>255) we are dividing individually

cv2.namedWindow('Method 3',cv2.WINDOW_NORMAL)
cv2.imshow('Method 3',matrix3.astype(np.uint8))  


#Method 4: using inbuilt function
img = cv2.imread('rgbcube.png',1)
matrix4 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.namedWindow('Method 3',cv2.WINDOW_NORMAL)
cv2.imshow('Method 3',matrix3.astype(np.uint8))  

cv2.waitKey(0)
cv2.destroyAllWindows()

