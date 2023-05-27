import numpy as np
import cv2   as cv

def getPoints(img_path):
    # Store the points
    points = []

    # Mouse callback function to store the points
    def fun(event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN and len(points) < 3:
            points.append(np.array([x,y]))
            print(f"X:{x}, Y:{y}")
            # Draw a circle at the point
            cv.circle(img, (x,y), 3, (0,0,255), -1)

    # Load the image
    img = cv.imread(img_path)
    # scale the image to 800x600    
    img = cv.resize(img, (800,600))
    # Create a window to display the image
    cv.namedWindow('image')
    # Set the mouse callback function to store the points
    cv.setMouseCallback('image', fun)

    # Display the image
    while len(points) < 3:
        cv.imshow('image', img)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # Get the image dimensions
    dims = img.shape
    # Load the camera matrix and distortion coefficients
    camera_matrix = np.loadtxt('camera_matrix.txt')
    # Focal length
    focal_length = camera_matrix[0,0]
    # The camera center is assumed to be at the center of the image
    C = np.array([dims[1]/2, dims[0]/2, 0])
    # The camera is assumed to be parallel to the image plane

    if len(points) == 3:
        # The points are converted to 3D points by appending 
        # the focal length of the camera and subtracting the camera center

        # Point in the 3D space
        p1 = np.append(points[0], focal_length) 
        p2 = np.append(points[1], focal_length)
        p3 = np.append(points[2], focal_length)

        points = [p1,p2,p3,C]

    cv.destroyAllWindows()
    return points, dims