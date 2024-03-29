#!/usr/bin/env python

import numpy as np
import cv2   as cv

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

print("RMS:", rms)
print("camera matrix:\n", camera_matrix)
print("distortion coefficients: ", dist_coefs.ravel())

# Store camera matrix and distortion coefficients in files for later use

# obtain camera parameters from the camera matrix
# focal length
#> hay que tomar la media con camera_matrix[0,0]
focal_length_p = camera_matrix[0,0]
# FOV horizontal
print(w)
hFov_p = 2 * np.arctan(w / (2 * focal_length_p))
# FOV vertical
print(h)
vFov_p = 2 * np.arctan(h / (2 * focal_length_p))

# print results
print("\n")
print("dimensions: \n", h, w)
print(f"Focal length: {focal_length_p} pix\n")
print("FOV horizontal: \n", hFov_p * 180 / np.pi)
print("FOV vertical: \n", vFov_p * 180 / np.pi)

np.savetxt('camera_matrix.txt', camera_matrix)
np.savetxt('dist_coefs.txt', dist_coefs)
np.savetxt('hFOV.txt', [hFov_p])
np.savetxt('vFOV.txt', [vFov_p])


