import cv2
import numpy as np
import math

img = cv2.imread('test.png',1)
n,m,l = img.shape
angle=int(input("Enter angle of rotation (+ for anticlockwise, - for clockwise): "))
theta = math.radians(angle)
s = math.sin(-theta)
c = math.cos(-theta)

# Method 1: Change position of each pixel using rotation formula:
# X+iY = (e^i0)*(x+iy)
new = np.full((1000,1000,l),0)
# We move from the centre to left and right
for i in range(int(-n/2),int(n/2)):
    for j in range(int(-m/2),int(m/2)):
        new[500+int(i*c+j*s),500+int(j*c-i*s),0] = img[int(n/2)+i,int(m/2)+j,0]
        new[500+int(i*c+j*s),500+int(j*c-i*s),1] = img[int(n/2)+i,int(m/2)+j,1]
        new[500+int(i*c+j*s),500+int(j*c-i*s),2] = img[int(n/2)+i,int(m/2)+j,2]

cv2.namedWindow('Rotated Image 1',cv2.WINDOW_NORMAL)
cv2.imshow('Rotated Image 1',new.astype(np.uint8))

center = (n/2,m/2)
# Method 2; Using in-built function
rotated_matrix = cv2.getRotationMatrix2D(center=center,angle=angle,scale=1)
rotated_img = cv2.warpAffine(src=img,M=rotated_matrix,dsize=(n,m))

cv2.namedWindow('Original Image',cv2.WINDOW_NORMAL)
cv2.imshow('Original Image',img.astype(np.uint8))

cv2.namedWindow('Rotated Image 2',cv2.WINDOW_NORMAL)
cv2.imshow('Rotated Image 2',rotated_img.astype(np.uint8))

cv2.waitKey(0)
cv2.destroyAllWindows()