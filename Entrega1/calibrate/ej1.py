#!/usr/bin/env python

import numpy as np
import cv2   as cv
from umucv.stream import autoStream

import sys
from glob import glob

files = glob(sys.argv[1])

square_size = 1
pattern_size = (9, 6)
pattern_points = np.zeros( (np.prod(pattern_size), 3), np.float32 )
pattern_points[:,:2] = np.indices(pattern_size).T.reshape(-1, 2)
pattern_points *= square_size

obj_points = []
img_points = []
h, w = 0, 0
for fn in files:
    print('processing %s...' % fn)
    img = cv.imread(fn, cv.IMREAD_GRAYSCALE)
    if img is None:
      print("Failed to load", fn)
      continue

    h, w = img.shape[:2]
    found, corners = cv.findChessboardCorners(img, pattern_size)
    if found:
        term = ( cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_COUNT, 30, 0.1 )
        cv.cornerSubPix(img, corners, (5, 5), (-1, -1), term)
    if not found:
        print('chessboard not found')
        continue
    img_points.append(corners.reshape(-1, 2))
    obj_points.append(pattern_points)

    print('ok')

rms, camera_matrix, dist_coefs, rvecs, tvecs = cv.calibrateCamera(obj_points, img_points, (w, h), None, None)

# Store camera matrix and distortion coefficients in files for later use
np.savetxt('camera_matrix.txt', camera_matrix)
np.savetxt('dist_coefs.txt', dist_coefs)

print("a) Calibración precisa\n")
print("RMS:", rms)
print("camera matrix:\n", camera_matrix)
print("distortion coefficients: \n", dist_coefs.ravel())
print("--------------------------------------------\n")


# obtain camera parameters from the camera matrix
# focal length
#> hay que tomar la media con camera_matrix[0,0] y camera_matrix[1,1]??
focal_length_p = camera_matrix[0,0]
# FOV horizontal
hFov_p = 2 * np.arctan(w / (2 * focal_length_p))
# FOV vertical
vFov_p = 2 * np.arctan(h / (2 * focal_length_p))


print("\n")
print("dimensions: \n", h, w)
print(f"Focal length: {focal_length_p} pix\n")
print("FOV horizontal: \n", hFov_p * 180 / np.pi)
print("FOV vertical: \n", vFov_p * 180 / np.pi)


# sleep the program until user presses a key 



print("b) Calibración aproximada\n")

from collections import deque

points = deque(maxlen=2)



##
img2 = cv.imread("./img/o.jpeg", cv.IMREAD_GRAYSCALE)
#cv.namedWindow("image")
#cv.setMouseCallback("image", fun)
#
#while True:
#    cv.imshow("image", img2)
#    if cv.waitKey(1) & 0xFF == ord("q"):
#        break
#
# get image size
h, w = img2.shape[:2]
print(f"Image size: {h}x{w}")

# marcar puntos en la imagen y medir la distancia en píxeles

# Medidas reales del objeto
object_h = 15  # cm
object_w = 3  # cm

# Medidas de la imagen
image_h = h  # pixels
image_w = w  # pixels

# Medidas de la imagen del objeto
image_object_h = 537  # pixels
image_object_w = 112  # pixels

# Distancia entre la cámara y el objeto
distance = 15  # cm

# Calculamos la focal length aproximada

focal_length_a = (image_object_h*distance)/object_h

# Calculamos el FOV vertical
vFov_a = 2*np.arctan(h/(2*focal_length_a))

# Calculamos FOV horizontal
hFov_a = 2*np.arctan(w/(2*focal_length_a))

# print fov en grados

print(f"Focal length: {focal_length_a} pix")
print("FOV vertical: {}º".format(vFov_a * 180 / np.pi))
print("FOV horizontal: {}º".format(hFov_a * 180 / np.pi))

print("--------------------------------------------\n")

print("c) Altura mínima de la cámara\n")

# Dimensiones de la pista
# Tomaremos el lado más largo como altura (depende de cómo esté orientada la cámara en la calibración anterior)
pista_h = 28*100  # cm
pista_w = 15*100  # cm

# Calculamos altura mínima para que la cámara vea toda la pista en horizontal
min_height_h = (pista_w/2) / np.tan(hFov_a/2)
print(min_height_h)
# En vertical
min_height_v = (pista_h/2) / np.tan(vFov_a/2)
print(min_height_v)

# altura mínima será la mayor de las dos
min_height = max(min_height_h, min_height_v)

print(f"Altura mínima: {min_height/100} m\n")

# load an image
img = cv.imread("./img/o.jpeg")

references = deque(maxlen=2)
sizes = deque(maxlen=4)


referencesSet = False

def fun(event, x, y, flags, param):
    global referencesSet
    if event == cv.EVENT_LBUTTONDOWN and not referencesSet:
        references.append((x,y))
        #horizontal angle
        h_angle = abs(x - w/2) * hFov_a / w
        #vertical angle
        v_angle = abs(y - h/2) * vFov_a / h
        print(f"Horizontal angle: {h_angle * 180 / np.pi}º")
        print(f"Vertical angle: {v_angle * 180 / np.pi}º")
        if len(references) == 2:
            referencesSet = True
            # calculate angle between the two points and the camera, using the camera matrix
            cy = camera_matrix[1,2]
            fy = camera_matrix[1,1]
            angle = np.arctan((references[0][1] - cy) / fy) - np.arctan((references[1][1] - cy) / fy)
            angle = angle * 180 / np.pi # convert to degrees
            print(f"Angle: {angle}º")

            # calculate distance between the two points
            distance = np.sqrt((references[0][0] - references[1][0])**2 + (references[0][1] - references[1][1])**2)
            print(f"Distance: {distance} pix")

        print(referencesSet)
    elif event == cv.EVENT_LBUTTONDOWN and referencesSet:
        sizes.append((x,y))
        print(f"Horizontal size 1: {abs(sizes[0][0]-sizes[1][0])}")
        print(f"Vertical size 1: {abs(sizes[0][1]-sizes[1][1])}")
        print(f"Horizontal size 2: {abs(sizes[2][0]-sizes[3][0])}")
        print(f"Vertical size 2: {abs(sizes[2][1]-sizes[3][1])}")


# open the image
cv.namedWindow("image")
cv.setMouseCallback("image", fun)

# imagen en bucle refrescando la imagen
while True:
    cv.imshow("image", img)
    # horizontal line in the middle
    cv.line(img, (0, int(h/2)), (w, int(h/2)), (0, 255, 0), 1)
    for p in references:
        cv.circle(img, p,3,(0,0,255),-1)
    for p in sizes:
        cv.circle(img, p,3,(0,255,255),-1)
    if cv.waitKey(1) & 0xFF == ord("q"):
        break




