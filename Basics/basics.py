
# 0(black) - 255(white) - for grayscale image 
# 0(min)  - 255(max) - for color image - in this case we need to specify all 3 R G B values
# pixels of image represented as elements of a matrix with each element having 3 dimensions - (x,y,color)
# we need 3 such elements to represent one pixel of a color image (RGB)

import cv2
import numpy as np
add_value = 25 
#Get image
matrix = np.full((600,600),255) #600x600 image with all pixels having color 255
img = cv2.imread('grayDog.jpeg',cv2.IMREAD_GRAYSCALE) #reading already present image

def incBrightness(value):
    global add_value 
    add_value = value
    

#Output image
cv2.namedWindow('Image',cv2.WINDOW_NORMAL)
cv2.imshow('Image',matrix.astype(np.uint8)) 
# by default it is signed int, but cv2 required unsigned int, so we are changing
# #Output second image (of puppy)
cv2.namedWindow('Image 2', cv2.WINDOW_NORMAL)
cv2.imshow('Image 2',img.astype(np.uint8))

cv2.createTrackbar('Brightness','Image 2',50,255,incBrightness)

while(1):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            #this iterates through every pixel of image
            if img[i,j] < 250 - add_value:
                img[i,j] += add_value
            else:
                img[i,j] = 255
    cv2.imshow('Image 2',img.astype(np.uint8))
    


cv2.waitKey(0) 
#if value inside is >0, it waits for that many milliseconds, otherwise its zero  => it waits forever
cv2.destroyAllWindows() #closes all windows