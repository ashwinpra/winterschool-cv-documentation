import cv2
import numpy as np

img = cv2.imread('test.png',1)

# Method 1: Manually

scale = int(input("Enter scaling factor (-ve if you want to shrink): "))
n,m,l = img.shape
if(scale<0):
    scale=-scale
    #Shrink the image
    new_n, new_m = int(n/scale), int (m/scale)
    matrix = np.full((new_n,new_m,l),0)
    for i in range(new_n):
        for j in range(new_m):
            sum_b,sum_g,sum_r=0,0,0
            for a in range(scale*i, scale*(i+1)):
                for b in range(scale*j, scale*(j+1)):
                    sum_b += img[a,b,0]
                    sum_g += img[a,b,1]
                    sum_r += img[a,b,2]
            sum_b/=(scale**2)
            sum_g/=(scale**2)
            sum_r/=(scale**2)
            matrix[i,j,0]=sum_b/(scale)
            matrix[i,j,1]=sum_g/(scale)
            matrix[i,j,2]=sum_r/(scale)
    # Method 2: Using in-built function
    # Here it is (m,n) instead of (n,m)
    new = cv2.resize(img,(new_m,new_n),interpolation = cv2.INTER_AREA)
    

else: 
    #Enlarge the image
    matrix = np.full((n*scale,m*scale,l),255)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for a in range(scale*i, scale*(i+1)):
                for b in range(scale*j, scale*(j+1)):
                    matrix[a,b] = img[i,j]
    # Method 2: Using in-built function
    new = cv2.resize(img,(m*scale,n*scale),interpolation = cv2.INTER_AREA)
    

cv2.namedWindow('New Image 1',cv2.WINDOW_NORMAL)
cv2.imshow('New Image 1',matrix.astype(np.uint8))
print(img.shape,matrix.shape,new.shape)
cv2.namedWindow('New Image 2',cv2.WINDOW_NORMAL)
cv2.imshow('New Image 2',new.astype(np.uint8))



cv2.waitKey(0)
cv2.destroyAllWindows()