import cv2
import numpy as np

img =cv2.imread('test.png',1)
cv2.namedWindow('Original Image',cv2.WINDOW_NORMAL)
cv2.imshow('Original Image',img.astype(np.uint8))

imgLine=img.copy()
ptA = (50,50)
ptB = (400,50)
# Draws a line through the image from ptA to ptB
cv2.line(imgLine,ptA,ptB,(255,0,0),thickness=5,lineType=cv2.LINE_AA)
# Different line types are possible
# Similarly we can draw circle using cv2.circle(src,center,radius,color,thickness,lineType)
# Giving thickness as -1 makes it a filled circle

cv2.namedWindow('Line',cv2.WINDOW_NORMAL)
cv2.imshow('Line',imgLine.astype(np.uint8))


cv2.waitKey(0)
cv2.destroyAllWindows()