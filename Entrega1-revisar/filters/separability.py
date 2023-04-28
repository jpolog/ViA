import cv2
import cv2 as cv

img = cv2.imread('./lena.png',0)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Create a Gaussian filter with a sigma of 10
kernel = cv2.getGaussianKernel(23, 10)

# Apply the filter in one direction
dst1 = cv2.sepFilter2D(img, -1, kernel, kernel)

# Apply the filter in both directions
dst2 = cv2.GaussianBlur(img, (23, 23), 10)

# Show the results
cv.imshow('original',img)

cv.imshow('Horizontal + Vertical',dst1)

cv.imshow('2D Gaussian',dst2)




# difference between the two results
dif = dst1-dst2
cv.imshow('difference', dif)

while True:
    if cv.waitKey(1) & 0xFF == ord('q'):
        break