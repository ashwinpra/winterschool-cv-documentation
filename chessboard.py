#make an 800x800 chessboard image
import cv2
import numpy as np
# we will first make it completely black, then loop through it 
# and change alternative 10x10 squares to white
chessboard = np.full((800,800),255)

for i in range(chessboard.shape[0]):
    for j in range(chessboard.shape[1]):
        if i%2 == j%2:
            chessboard[i, j] = 0

cv2.namedWindow('Image',cv2.WINDOW_NORMAL)
cv2.imshow('Image',chessboard.astype(np.uint8))

cv2.waitKey(0)
cv2.destroyAllWindows()