import cv2
import numpy as np

img = cv2.imread('test.png',1)
n,m,l = img.shape

tx = int(input("Enter translation value in x: "))
ty = int(input("Enter translation value in y: "))

matrix = np.array([[1,0,tx],[0,1,ty]],dtype=np.float32)

new = cv2.warpAffine(src=img,M=matrix,dsize=(n,m))

cv2.namedWindow('Original Image',cv2.WINDOW_NORMAL)
cv2.imshow('Original Image',img.astype(np.uint8))

cv2.namedWindow('Translated Image',cv2.WINDOW_NORMAL)
cv2.imshow('Translated Image',new.astype(np.uint8))

cv2.waitKey(0)
cv2.destroyAllWindows()
