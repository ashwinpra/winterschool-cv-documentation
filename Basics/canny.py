import cv2 
import numpy as np

vid = cv2.VideoCapture(0) 

while 1:
    ret, frame = vid.read() #ret is True or False
    cv2.imshow('frame', frame)
    
    # Before canny we will do HSV thresholding for better result 
    # HSV = Hue Saturation Value
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # define range of color in HSV
    # change the ranges according to your need
    lower,upper = np.array([30,150,50]),np.array([255,255,100])

    # Threshold the HSV image with this range
    mask = cv2.inRange(hsv, lower, upper)

    # This returns white for all areas that lies in this range, and black for those that lie outside
    # So we do bitwise AND of this with og image
    res = cv2.bitwise_and(frame,frame,mask=mask)

    edges = cv2.Canny(res, 100, 200) # img, threshold value 1, threshold value 2
    cv2.imshow('edges', edges)

    # cv2.waitKey returns a 32 bit binary number which corresponds to pressed key 
    # & (bitwise operator) extracts last 8 bits, and does & with 11111111 (0xFF)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


vid.release()
cv2.destroyAllWindows()