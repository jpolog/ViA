#!/usr/bin/env python

import numpy as np
import cv2   as cv
from anglesFunction import get_image_dimensions_points_and_angle
import math

# calculate projection in y=0 plane
def get_projection(points):
    ret = []
    for p in points:
        ret.append(np.array([p[0],p[2]])) # se suprime la coordenada y
    return ret

# calculate midpoint of two points
def get_midpoint(p1,p2):
    return np.array([(p1[0]+p2[0])/2,(p1[1]+p2[1])/2])

# calculate distance between two points
def get_distance(p1,p2):
    return np.linalg.norm(p1-p2)

# calculate angle between two points and a third point
def get_angle(points, ref):
    v1 = points[0]-ref
    v2 = points[1]-ref
    return np.arccos(np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)))
    # in degrees
    


# get radius of circle from two points and angle of the arc between them
def get_radius(p1,p2, angle):
    d = get_distance(p1,p2)
    return (d/2)/np.sin(angle)
    #return np.sqrt(1/4*((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)*(1/np.sin(angle/2))**2)

# get center of the circle from two points and angle of the arc between them
def get_center(p1,p2, angle):
    # get radius
    r = get_radius(p1,p2, angle)
    # get midpoint
    m = get_midpoint(p1,p2)
    # from midpoint, go down by r*cos(angle/2)
    return np.array([m[0],m[1]+r*np.cos(angle)])

def get_intersections(x0, y0, r0, x1, y1, r1):
    # circle 1: (x0, y0), radius r0
    # circle 2: (x1, y1), radius r1

    d=math.sqrt((x1-x0)**2 + (y1-y0)**2)
    
    # non intersecting
    if d > r0 + r1 :
        return None
    # One circle within other
    if d < abs(r0-r1):
        return None
    # coincident circles
    if d == 0 and r0 == r1:
        return None
    else:
        a=(r0**2-r1**2+d**2)/(2*d)
        h=math.sqrt(r0**2-a**2)
        x2=x0+a*(x1-x0)/d   
        y2=y0+a*(y1-y0)/d   
        x3=x2+h*(y1-y0)/d     
        y3=y2-h*(x1-x0)/d 

        x4=x2-h*(y1-y0)/d
        y4=y2+h*(x1-x0)/d
        
        return ([x3, y3], [x4, y4])

# calculate the camera position
def get_camera_position(p1,p2,p3,c):
    # calculate projection in y=0 plane
    p1,p2,p3,c = get_projection([p1,p2,p3,c])
    # calculate midpoint of points in pairs
    m1 = get_midpoint(p1,p2)
    m2 = get_midpoint(p2,p3)
    # calculate distance between points in pairs
    d1 = get_distance(p1,p2)
    d2 = get_distance(p2,p3)
    # calculate angle between points in pairs
    a1 = get_angle([p1,p2],c)
    a2 = get_angle([p2,p3],c)
    # get center of each circle
    c1 = get_center(p1,p2,a1)
    c2 = get_center(p2,p3,a2)

    # Circle 1 defined by c1 and r1
    circle1 = (c1[0],c1[1],get_radius(p1,p2,a1))
    # Circle 2 defined by c2 and r2
    circle2 = (c2[0],c2[1],get_radius(p2,p3,a2))

    # get intersections of the two circles
    intersections = get_intersections(circle1[0],circle1[1],circle1[2],circle2[0],circle2[1],circle2[2])

    # print each step
    print('p1: ',p1)
    print('p2: ',p2)
    print('p3: ',p3)
    print('c: ',c)
    print('m1: ',m1)
    print('m2: ',m2)
    print('d1: ',d1)
    print('d2: ',d2)
    print('a1 (deg): ',a1*180/np.pi)
    print('a2 (deg): ',a2*180/np.pi)
    print('c1: ',c1)
    print('c2: ',c2)
    print('circle1: ',circle1)
    print('circle2: ',circle2)
    print('intersections: ',intersections)


get_camera_position(np.array([-10,16.57,-10]),np.array([2.77,11.43,-10]),np.array([13.29,16.63,-10]),np.array([0,15,15]))



