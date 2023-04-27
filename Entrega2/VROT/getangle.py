import numpy as np

def getAngle(img, p1, p2):
    
    points = [p1, p2]

    # Get the image dimensions
    dims = img.shape
    # Load the camera matrix and distortion coefficients
    camera_matrix = np.loadtxt('camera_matrix.txt')
    # Focal length
    focal_length = camera_matrix[0,0]
    # The camera center is assumed to be at the center of the image
    C = np.array([dims[1]/2, dims[0]/2, 0])
    # The camera is assumed to be parallel to the image plane


    # Calculate the angle between the two points
    angle = None
    if len(points) == 2:
        # The two points are converted to 3D points by appending the focal length of the camera and subtracting the camera center
        # Vector in the 3D space
        p1 = np.append(points[0], focal_length) - C
        p2 = np.append(points[1], focal_length) - C
        # Calculate the angle using the dot product (producto escalar)
        angle = np.arccos(np.dot(p1, p2) / (np.linalg.norm(p1) * np.linalg.norm(p2)))
        # in degrees
        angle = angle * 180 / np.pi
        
        return angle
