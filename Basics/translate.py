import cv2
import numpy as np

img = cv2.imread('test.png',0)
n,m = img.shape

tx = int(input("Enter translation value in x: "))
ty = int(input("Enter translation value in y: "))

new_matrix = np.full((n,m),0)
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if (i>tx and j>ty):
            new_matrix[i,j] = img[i-tx,j-ty]

matrix = np.array([[1,0,tx],[0,1,ty]],dtype=np.float32)

new = cv2.warpAffine(src=img,M=matrix,dsize=(n,m))

cv2.namedWindow('Original Image',cv2.WINDOW_NORMAL)
cv2.imshow('Original Image',img.astype(np.uint8))

cv2.namedWindow('Translated Image',cv2.WINDOW_NORMAL)
cv2.imshow('Translated Image',new.astype(np.uint8))

cv2.namedWindow('Translated Image 2',cv2.WINDOW_NORMAL)
cv2.imshow('Translated Image 2',new_matrix.astype(np.uint8))

cv2.waitKey(0)
cv2.destroyAllWindows()
