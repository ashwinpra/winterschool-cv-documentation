import cv2 
import numpy as np

vid = cv2.VideoCapture(0) 

while 1:
    ret, frame = vid.read() #ret is True or False
    cv2.imshow('frame', frame)
    # cv2.waitKey returns a 32 bit binary number which corresponds to pressed key 
    # & (bitwise operator) extracts last 8 bits, and does & with 11111111 (0xFF)
    # 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


vid.release()
cv2.destroyAllWindows()