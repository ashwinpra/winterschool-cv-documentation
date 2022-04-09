import cv2
import numpy as np 

vid = cv2.VideoCapture(0) # 0 signifies webcam

if not vid.isOpened():
	print("Cannot open camera")
	exit()

while 1:
	ret, frame = vid.read() # Capturing frame-by-frame
	print(ret)
	if not ret:
		print("Cant receive frame. Exiting...")
		break

	# Operation on frame goes here
	# cv2.imshow(processed_frame)

	if cv2.waitKey(1) == ord('q'):
		break