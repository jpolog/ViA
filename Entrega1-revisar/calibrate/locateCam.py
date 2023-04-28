#!/usr/bin/env python

import numpy as np
import cv2   as cv
from anglesFunction import get_image_dimensions_points_and_angle

# locate the camera from two points in the image and the angle between them and the camera

# Get the image dimensions, points and angle
dims, points, angle = get_image_dimensions_points_and_angle('./mylogitech/20150309-091608.png')

# Draw a 2D plane

