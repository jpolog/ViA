#!/usr/bin/env python

# Ejercicio 1.d)

import numpy as np
import cv2   as cv



# Store the points
points = []

# Mouse callback function to store the points
def fun(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN and len(points) < 2:
        points.append(np.array([x,y]))
        print(f"X:{x}, Y:{y}")
        # Draw a circle at the point
        cv.circle(img, (x,y), 3, (0,0,255), -1)

# Load the image
img = cv.imread('./mylogitech/20150309-091608.png', cv.IMREAD_GRAYSCALE)
# Create a window to display the image
cv.namedWindow('image')
# Set the mouse callback function to store the points
cv.setMouseCallback('image', fun)

# Display the image
while True:
    cv.imshow('image', img)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break


# Get the image dimensions
dims = img.shape
# Load the camera matrix and distortion coefficients
camera_matrix = np.loadtxt('camera_matrix.txt')
dist_coefs = np.loadtxt('dist_coefs.txt')
# Focal length
focal_length = camera_matrix[0,0]
# The camera center is assumed to be at the center of the image
C = np.array([dims[1]/2, dims[0]/2, 0])
# The camera is assumed to be parallel to the image plane


# Calculate the angle between the two points
if len(points) == 2:
    # The two points are converted to 3D points by appending the focal length of the camera and subtracting the camera center
    # Vector in the 3D space
    p1 = np.append(points[0], focal_length) - C
    p2 = np.append(points[1], focal_length) - C
    print(f"p1: {p1}, p2: {p2}")
    # Calculate the angle using the dot product (#> producto escalar)
    angle = np.arccos(np.dot(p1, p2) / (np.linalg.norm(p1) * np.linalg.norm(p2)))
    # in degrees
    angle = angle * 180 / np.pi
    print(f"Angle: {angle}")


cv.destroyAllWindows()

    
