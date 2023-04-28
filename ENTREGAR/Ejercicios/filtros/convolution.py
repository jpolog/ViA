import numpy as np
import cv2   as cv
import time

# convolution algorithm with loops
def convolve2d(image, kernel):
    # get shapes of image and kernel
    image_row, image_col = image.shape
    kernel_row, kernel_col = kernel.shape

    # output shape the same as original image
    output = np.zeros([image_row,image_col])

    # convolution operation
    # in the edges of the image, the kernel is not fully inside the image
    # in this case, we take 0 for the values outside the image

    # edge handling with extend method
    # edge to apply the kernel of the image with the value of the nearest pixel
    image = np.pad(image, ((kernel_row // 2, kernel_row // 2), (kernel_col // 2, kernel_col // 2)), 'edge')

    # count the time
    start = time.time()
    # convolution operation
    for i in range(image_row):
        for j in range(image_col):
            # if the border
            # element-wise multiplication of the kernel and the image
            output[i, j] = (kernel * image[i: i + kernel_row, j: j + kernel_col]).sum()
    stop = time.time()
    convolution_time = stop - start
    return output, convolution_time


# convolution algorithm example
img = cv.imread('./lena.png ',0)
kernel = np.array([[0.00376508, 0.01501909, 0.02379215, 0.01501909, 0.00376508],
       [0.01501909, 0.05991246, 0.09490781, 0.05991246, 0.01501909],
       [0.02379215, 0.09490781, 0.15034264, 0.09490781, 0.02379215],
       [0.01501909, 0.05991246, 0.09490781, 0.05991246, 0.01501909],
       [0.00376508, 0.01501909, 0.02379215, 0.01501909, 0.00376508]])

dst, time_1 = convolve2d(img,kernel)
print('convolution time: ', time_1)

# apply filter with opencv
start = time.time()
opencv = cv.filter2D(img,-1,kernel)
stop = time.time()
time_2 = stop-start
print('opencv time: ', time_2)

# calculate % faster in time
perc = time_1/time_2
print('opencv is %f times faster than convolution' % perc)


# np array to image
dst = np.uint8(dst)
cv.imshow('original',img)
cv.imshow('convolution',dst)
cv.imshow('opencv',opencv)






cv.waitKey(0)
