#make an 800x800 chessboard image
import cv2
import numpy as np
# we will first make it completely black, then loop through it 
# and change alternative 10x10 squares to white
chessboard = np.full((800,800),0)

for i in range(0,800,100):
    for j in range(0,800,100):
        for k1 in range(i,i+100):
            for k2 in range(j,j+100):
                if(i%200==j%200):
                    chessboard[k1,k2]=255

cv2.namedWindow('Image',cv2.WINDOW_NORMAL)
cv2.imshow('Image',chessboard.astype(np.uint8))

cv2.waitKey(0)
cv2.destroyAllWindows()
