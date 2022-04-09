import cv2
import numpy as np

img1 = cv2.imread('blob.png',0)
#Blurring it first
img1 = cv2.blur(img1,(5,5)) #(5,5) is kernel size
n,m = img1.shape
img2 = cv2.imread('blob.png',0)
img2 = cv2.blur(img1,(5,5)) #(5,5) is kernel size
n,m = img2.shape


# Then applying thresholdning: 
# ie, all pixel values >127 are converted to 255
# all pixel values <127 are converted to  

for i in range(img1.shape[0]):
    for j in range(img1.shape[1]):
        if img1[i,j]>127:
            img1 [i,j] = 255
        else:
            img1[i,j] = 0 
for i in range(img2.shape[0]):
    for j in range(img2.shape[1]):
        if img2[i,j]>127:
            img2 [i,j] = 255
        else:
            img2[i,j] = 0 
#Now we have a binary image, so we can find blobs in it

# we check through the image with dfs 
# we will use recursion, and check all neighbours of (x,y)
# if any neighbour has 255 value, then it will do checkDFS(neighbour)

def checkDFS(x,y):
    img1[x,y] = 127 #to imply we have visited that pixel
    #checking every child of (x,y)
    if x>0 and img1[x-1,y]==255:
        checkDFS(x-1,y)
    if x<n-1 and img1[x+1,y]==255:
        checkDFS(x+1,y)
    if y>0 and img1[x,y-1]==255:
        checkDFS(x,y-1)
    if y<m-1 and img1[x,y+1]==255:
        checkDFS(x,y+1)
    if x>0 and y>0 and img1[x-1,y-1]==255:
        checkDFS(x-1,y-1)
    if x>0 and y<m-1 and img1[x-1,y+1]==255:
        checkDFS(x-1,y+1)
    if x<n-1 and y>0 and img1[x+1,y-1]==255:
        checkDFS(x+1,y-1)
    if x<n-1 and y<m-1 and img1[x+1,y+1]==255:
        checkDFS(x+1,y+1)

queue = []
def checkBFS(x,y):  
    global queue
    img2[x,y] = 127 #to imply we have visited that pixel
    #checking every child of (x,y)
    if x>0 and img2[x-1,y]==255:
        queue.append((x-1,y))
    if x<n-1 and img2[x+1,y]==255:
        queue.append((x+1,y))
    if y>0 and img2[x,y-1]==255:
        queue.append((x,y-1))
    if y<m-1 and img2[x,y+1]==255:
        queue.append((x,y+1))
    if x>0 and y>0 and img2[x-1,y-1]==255:
        queue.append((x-1,y-1))
    if x>0 and y<m-1 and img2[x-1,y+1]==255:
        queue.append((x-1,y+1))
    if x<n-1 and y>0 and img2[x+1,y-1]==255:
        queue.append((x+1,y-1))
    if x<n-1 and y<m-1 and img2[x+1,y+1]==255:
        queue.append((x+1,y+1))

    


countDFS = 0
countBFS = 0
for i in range(img1.shape[0]):
    for j in range(img1.shape[1]):
        if img1[i,j]==255:
            countDFS+=1 #keeps track of number of blobs
            checkDFS(i,j)
for i in range(img1.shape[0]):
    for j in range(img1.shape[1]):
        if img2[i,j]==255:
            countBFS+=1
            checkBFS(i,j)
            while(len(queue)!=0):
                x,y = queue.pop(0)
                if(img2[x,y]==255):
                    checkBFS(x,y) 
                    

print(countDFS,countBFS)