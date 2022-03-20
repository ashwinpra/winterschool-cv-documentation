import cv2
import numpy as np

#Get image
matrix = np.full((600,600),255) #600x600 image with all pixels having color 255
img = cv2.imread('gray.jpeg',cv2.IMREAD_GRAYSCALE) #reading already present image

print(img.shape)
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        #any command (this iterates through every pixel of image)
        pass

# 0(black) - 255(white) - for grayscale image 
# 0(min)  - 255(max) - for color image - in this case we need to specify all 3 R G B values
# pixels of image represented as a matrix with each element having 3 dimensions - (x,y,color)
# we need 3 such elements to represent one pixel of a color image (RGB)


#Output image
cv2.namedWindow('Image',cv2.WINDOW_NORMAL)
cv2.imshow('Image',matrix.astype(np.uint8)) 
#we have to make sure the matrix has all integer entries, hence we are changing type
cv2.namedWindow('Image 2', cv2.WINDOW_NORMAL)
cv2.imshow('Image 2',img.astype(np.uint8))

cv2.waitKey(0) #when 0 is pressed, window is closed
cv2.destroyAllWindows() #closes all windows