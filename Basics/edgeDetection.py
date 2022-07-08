import cv2
import numpy as np

# Sobel filter is used to get gradient in each direction 

# syntax: img1 = cv2.Sobel(img, ddepth, dx, dy, ksize)
# dx and dy: direction in which you want to take gradient 
# Ksize: size of filter (kernel) which could be 3,5,7
# Ddepth: taken as CV_64F

# Better method: Canny Edge Detection
# Involves 4 steps of processing: 
# 1. Noise Reduction (5x5 Gaussian Blur)
# 2. Finding Intensity Gradient of Image (Sobel)
# 3. Non-Maximum Suppression 
# 4. Hysteresis Thresholding 
# Rather than just finding intensity gradient, it does all 4 of these steps

# syntax: img1 = cv2.Canny(img,maxVal,minVal)
# maxVal and minVal are values used in Hysteresis Thresholding; generally taken in ratio 2:1 or 3:1

# Hough Line Transformation: 
# Used to detect curves in *binary* images after they have undergoen edge detection (edges depicted as white)
# It works by analysing each pixel in the image and plotting them in a (theta,r) space (Hough plane) if they are white
# The nuber of intersecton points gives number of unique lines and coordinates of each such point guves parameters of each of those lines 

# syntax: cv.HoughLines(img,rho,theta,threshold)
# img: 8bit, single-channel binary source image
# rho: resolution of parameter r 
# theta: resolution of parameter theta (in radians)
# threshold: minimum number of intersections to "detect" a line
# Read docs for more information

# Contour: A curve joining all the continuous points (along a boundary), having same color/intensity 
# It is also found on *binary* images
# Before finding countours, threshold / canny edge detection is done
# Finding contours is like finding white object from black background, so that colors should be maintained
# using cv2.findContours() you can get a list of all the contours and their hierarchy
# Then you can use cv2.drawContours() to draw those
# Read docs for more information 

# Contour hierarchy:
# If a contour is completely within the other, it is a child, otherwise it is a sibling 

# Template Matching: 
# A method for searching and finding the location of a template image in a larger image
# cv2.matchTemplate(image, template, method) is used for this purpose
# It returns a grayscale image, where each pixel denotes how much the nbhd of that pixel matches with template
# min_val, max_val, min_loc, max_loc = cv2.minMaxLoc() is isued to find maximum score
# min and max loc are x and y coordinates of top left corner
# min and max val are the length and breadth respectively

# Detecting Multiple Objects 
# Use matchTempalte to give grayscale image
# Use thresholding to get best output

# Limitations: 
# This method preserves the orientation of the image, so it wont work if the image is rotated

# Corner Detection: 
# We check the intensity and if there is large intensity variation in all directons, it is a corner\\

# Harris Corner Detection: 
# syntax: img = cv2.cornerHaris(src,blocksize,ksize,k)
# blockSize: nbhd size
# K: Harris detector free parameter
# It is just taken as cv2.cornerHarris(img,2,3,0.04)
# This returns a grayscale image with "R" values at each point 
# Threshold this (0.01*max) to detect corner points 

# Dijkstra Algorithm
# https://medium.com/basecs/finding-the-shortest-path-with-a-little-help-from-dijkstra-613149fbdc8e
# Basically finds the shortest path when the path is weighted and you want to minimize the weight\

# Image optimisation 

# Histogram equalisation