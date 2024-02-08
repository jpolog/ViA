# check the cascading property of the gaussian filter 
# the key is that the convolution between two gaussian filters is also a gaussian filter 

import numpy as np
import cv2 as cv
#load the image
img = cv.imread('./lena.png',0)
# apply a gaussian filter with sigma = X to the previous result
sx = 2.5
g1 =  cv.GaussianBlur(img,(0,0),sx)
# apply a gaussian filter with sigma = Y to the previous result
sy = 4
g2 =  cv.GaussianBlur(g1,(0,0),sy)

# apply a gaussian filter with sigma = sqrt(X²+Y²) to the original image
g3 =  cv.GaussianBlur(img,(0,0),np.sqrt(sx**2+sy**2))

# show the results
cv.imshow('original',img)
cv.imshow('cascading 2-step',g2)
cv.imshow('1-step',g3)

# difference between the two results
dif = g2-g3
cv.imshow('difference', dif)

# % error
print('error: ', np.sum(dif)/np.sum(g3))


cv.waitKey(0)

