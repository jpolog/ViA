#!/usr/bin/env python

import sys
import numpy as np
import matplotlib.pyplot as plt
import math

from getPoints import getPoints

imagePath = 'locate.jpg'
# para el ejemplo teórico ejecutar con:
# ./locateCam.py 15,30 35,45 40,35 36.39 19.42

#get points from args and the angles 
# the format is x1,y1 x2,y2 x3,y3 a1 a2
def get_args():
    # get points
    p1 = np.array([float(sys.argv[1].split(',')[0]),float(sys.argv[1].split(',')[1])])
    p2 = np.array([float(sys.argv[2].split(',')[0]),float(sys.argv[2].split(',')[1])])
    p3 = np.array([float(sys.argv[3].split(',')[0]),float(sys.argv[3].split(',')[1])])
    # get angles
    a1 = float(sys.argv[4])
    a2 = float(sys.argv[5])
    return [p1,p2,p3],[a1,a2]



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
    


# get distance from the midpoint of two points to the center of the circle
def get_d_from_center(p1,p2, angle):
    d = get_distance(p1,p2)
    return (d/2)/np.sin(angle)

# get center of the circle from two points and angle of the arc between them
def get_center_image(p1,p2, angle):
    # get radius
    r = get_d_from_center(p1,p2, angle)
    # get midpoint
    m = get_midpoint(p1,p2)
    # from midpoint, go down by r*cos(angle)
    return np.array([m[0],m[1]-r*np.cos(angle)])

def get_center_manual(p1,p2, angle):
    # Get midpoint
    m = get_midpoint(p1, p2)
    # Calculate the perpendicular distance from the midpoint to the segment
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    d = get_d_from_center(p1, p2, angle) * math.cos(angle)
    # Calculate the angle bisector
    bisector_angle = math.atan2(dy, dx) + math.pi / 2
    # Calculate the coordinates of the center point
    center_x = m[0] + d * math.cos(bisector_angle)
    center_y = m[1] + d * math.sin(bisector_angle)
    return np.array([center_x, center_y])

def get_intersections(x0, y0, r0, x1, y1, r1):
    # circle 1: (x0, y0), radius r0
    # circle 2: (x1, y1), radius r1

    d=math.sqrt((x1-x0)**2 + (y1-y0)**2)
    
    # non intersecting
    if d > r0 + r1 :
        print("non intersecting")
        return None
    # One circle within other
    if d < abs(r0-r1):
        print("One circle within other")
        return None
    # coincident circles
    if d == 0 and r0 == r1:
        print("coincident circles")
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
def get_camera_position_image(points):
    p1,p2,p3,c = points[0],points[1],points[2],points[3]
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
    o1 = get_center_image(p1,p2,a1)
    o2 = get_center_image(p2,p3,a2)

    # Circle 1 defined by o1 and r1
    circle1 = (o1[0],o1[1],get_d_from_center(p1,p2,a1))
    # Circle 2 defined by o2 and r2
    circle2 = (o2[0],o2[1],get_d_from_center(p2,p3,a2))

    # get intersections of the two circles
    intersections = get_intersections(circle1[0],circle1[1],circle1[2],circle2[0],circle2[1],circle2[2])
    if intersections is None:
        sys.exit(1)
        
    # get intersection that is different from p2
    if get_distance(intersections[0],p2) > get_distance(intersections[1],p2):
        c = intersections[0]
    else:
        c = intersections[1]

    return [p1,p2,p3,c], [m1,m2], [d1,d2], [a1,a2], [o1,o2], [circle1, circle2], intersections


def get_camera_position_manual(points,angles):
    p1,p2,p3 = points[0],points[1],points[2]
    a1,a2 = angles[0],angles[1]
    # calculate midpoint of points in pairs
    m1 = get_midpoint(p1,p2)
    m2 = get_midpoint(p2,p3)
    # calculate distance between points in pairs
    d1 = get_distance(p1,p2)
    d2 = get_distance(p2,p3)
    # get center of each circle
    o1 = get_center_manual(p1,p2,a1)
    o2 = get_center_manual(p2,p3,a2)

    # Circle 1 defined by o1 and r1
    circle1 = (o1[0],o1[1],get_d_from_center(p1,p2,a1))
    # Circle 2 defined by o2 and r2
    circle2 = (o2[0],o2[1],get_d_from_center(p2,p3,a2))

    # get intersections of the two circles
    intersections = get_intersections(circle1[0],circle1[1],circle1[2],circle2[0],circle2[1],circle2[2])
    if intersections is None:
        sys.exit(1)

    # get intersection that is different from p2
    if get_distance(intersections[0],p2) > get_distance(intersections[1],p2):
        c = intersections[0]
    else:
        c = intersections[1]

    return [p1,p2,p3,c], [m1,m2], [d1,d2], [a1,a2], [o1,o2], [circle1, circle2], intersections
    


def show_results(points, midpoints, distances, angles, centers, circles, intersections):
    p1,p2,p3,c = points[0],points[1],points[2],points[3]
    m1,m2 = midpoints[0],midpoints[1]
    d1,d2 = distances[0],distances[1]
    a1,a2 = angles[0],angles[1]
    o1,o2 = centers[0],centers[1]
    circle1, circle2 = circles[0],circles[1]
    # print each value
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
    print('o1: ',o1)
    print('o2: ',o2)
    print('circle1: ',circle1)
    print('circle2: ',circle2)
    print('intersections: ',intersections)
    print('camera position: ',c)

    # represent the points and circles in a plot of dimensions enough to see all the elements
    fig, ax = plt.subplots()
    ax.set_aspect(1)
    ax.add_artist(plt.Circle((circle1[0], circle1[1]), circle1[2], color='r', fill=False))
    ax.add_artist(plt.Circle((circle2[0], circle2[1]), circle2[2], color='b', fill=False))
    ax.plot([p1[0],p2[0],p3[0],c[0]],[p1[1],p2[1],p3[1],c[1]],'ro')
    ax.plot([m1[0],m2[0]],[m1[1],m2[1]],'bo')
    ax.plot([o1[0],o2[0]],[o1[1],o2[1]],'go')
    

    # label the points and circles
    ax.annotate('p1', (p1[0],p1[1]))
    ax.annotate('p2', (p2[0],p2[1]))
    ax.annotate('p3', (p3[0],p3[1]))
    ax.annotate('c', (c[0],c[1]))
    ax.annotate('m1', (m1[0],m1[1]))
    ax.annotate('m2', (m2[0],m2[1]))
    ax.annotate('o1', (o1[0],o1[1]))
    ax.annotate('o2', (o2[0],o2[1]))
    
    # set the limits of the plot
    # calculate dimensions so that all the elements are visible with a margin
    # including the circles
    x_min = min(p1[0],p2[0],p3[0],c[0],m1[0],m2[0],o1[0],o2[0],circle1[0]-circle1[2],circle2[0]-circle2[2]) - 10
    x_max = max(p1[0],p2[0],p3[0],c[0],m1[0],m2[0],o1[0],o2[0],circle1[0]+circle1[2],circle2[0]+circle2[2]) + 10
    y_min = min(p1[1],p2[1],p3[1],c[1],m1[1],m2[1],o1[1],o2[1],circle1[1]-circle1[2],circle2[1]-circle2[2]) - 10
    y_max = max(p1[1],p2[1],p3[1],c[1],m1[1],m2[1],o1[1],o2[1],circle1[1]+circle1[2],circle2[1]+circle2[2]) + 10
    ax.set_xlim([x_min,x_max])
    ax.set_ylim([y_min,y_max])
    


    plt.show()


# Main execution

# if number of arguments is 1, then is the name of the image
# ---> image mode
if len(sys.argv) == 2:
    # read points from image
    image = sys.argv[1]
    points,dims = getPoints(image)
    # calculate projection in y=0 plane
    points = get_projection(points)
    points, midpoints, distances, angles, centers, circles, intersections = get_camera_position_image(points)
elif len(sys.argv) == 6:
    # if number of arguments is 6, then are the 2D coordinates of the points (p1,p2,p3) and the 2 angles (a1,a2)
    # ---> manual mode
    points, angles = get_args()
    #angles to radians
    angles = [a*np.pi/180 for a in angles]
    points, midpoints, distances, angles, centers, circles, intersections = get_camera_position_manual(points,angles)
else:
    if len(sys.argv) < 6:
        print('Usage: python3 locateCamp.py x1,y1 x2,y2 x3,y3 a1 a2')
        print('or')
        print('Usage: python3 locateCam.py /path/to/image')
        sys.exit(1)


show_results(points, midpoints, distances, angles, centers, circles, intersections)





