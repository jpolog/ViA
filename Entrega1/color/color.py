import copy
import cv2 as cv
import numpy as np

############################################################
# Functions to extract and filter the contours of the mask #
############################################################

def extractContours(g):
    contours, _ = cv.findContours(
        g.copy(), cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)[-2:]
    contours = [c.reshape(-1, 2) for c in contours]
    contours = sorted(contours, key=cv.contourArea, reverse=True)
    return contours

# Para eliminar contornos claramente erróneos


# razonable si su área es mayor que 10**2 y menor que 100**2
def razonable(c):
    return 100**2 >= cv.contourArea(c) >= 10**2

# esta función nos indica si el contorno se recorre en sentido horario o
# antihorario. La función findContours
# recorre en sentido antihorario las regiones True de la máscara binaria,
# (que con black=True serán las manchas oscuras) y en sentido horario
# los agujeros y las regiones claras.
def orientation(x):
    return cv.contourArea(x.astype(np.float32), oriented=True) >= 0


############################################################
# Functions to calculate the average color of the selected #
# pixels and to count the objects                          #
############################################################

def calculate_range(average_color, h, s, v):

    # Define the range of colors to look for (in this case, colors within 20 of the average color in the HSV color space)
    lower_color_range = np.array(
        [average_color_hsv[0] - h, average_color_hsv[1] - s, average_color_hsv[2] - v])
    upper_color_range = np.array(
        [average_color_hsv[0] + h, average_color_hsv[1] + s, average_color_hsv[2] + v])

    return lower_color_range, upper_color_range


def count_objects(image, average_color):

    # to revert the drawing of the contours
    img_cache = copy.deepcopy(image)

    cv.namedWindow('Mask')
    # Trackbars to refine the mask
    # create trackbars for color change
    cv.createTrackbar('Hue', 'Mask', 0, 179, lambda x: None)
    cv.createTrackbar('Saturation', 'Mask', 0, 255, lambda x: None)
    cv.createTrackbar('Value', 'Mask', 0, 255, lambda x: None)

    # set default value for trackbars
    cv.setTrackbarPos('Hue', 'Mask', 20)
    cv.setTrackbarPos('Saturation', 'Mask', 60)
    cv.setTrackbarPos('Value', 'Mask', 60)

    num_objects = 0  # store the number of objects

    # recalculate the mask if trackbar changes
    while True:
        # get current positions of four trackbars
        h = cv.getTrackbarPos('Hue', 'Mask')
        s = cv.getTrackbarPos('Saturation', 'Mask')
        v = cv.getTrackbarPos('Value', 'Mask')

        lower_color_range, upper_color_range = calculate_range(average_color, h, s, v)
        mask = cv.inRange(hsv_image, lower_color_range, upper_color_range)
        cv.imshow('Mask', mask)

        # Apply the mask to the original image
        result = cv.bitwise_and(image, image, mask=mask)

        # invert mask to get the black shapes in white background
        mask = cv.bitwise_not(mask)

        # Extraemos los contornos de la máscara
        contours = extractContours(mask)
        # Eliminamos los contornos erróneos
        contours = [c for c in contours if razonable(c) and orientation(c)]

        # print if the number of objects has changed
        if num_objects != len(contours):
            num_objects = len(contours)
            print('Number of objects:', num_objects)
            print('\n################################\n')

            # to revert the drawing of the contours
            image = copy.deepcopy(img_cache)

        # Dibujamos los contornos
        cv.drawContours(image, contours, -1, (0, 255, 0), 2)

        # Display the result
        cv.imshow('Color Mask', result)
        # Mostramos la imagen con los contornos
        cv.imshow('Contours', image)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break


# Load the image
image = cv.imread('img.jpg')

selected_pixels = []

# Create a window to display the image and set the mouse callback function
cv.namedWindow('image')
cv.setMouseCallback('image', lambda event, x, y, flags,
                    param: on_mouse_click(event, x, y, flags, param, image))

# Function to handle mouse clicks


def on_mouse_click(event, x, y, flags, param, image):
    # Check if left button is clicked
    if event == cv.EVENT_LBUTTONDOWN:
        # Get the color of the pixel
        pixel_color = image[y, x]
        selected_pixels.append(pixel_color)
        print('Pixel color:', pixel_color)


while True:
    cv.imshow('image', image)

    # Wait for a mouse click to select the 3 pixels
    if len(selected_pixels) == 3:
        # Calculate the average color of the 3 pixels
        average_color = np.average(selected_pixels, axis=0)
        print(f'Average color: {average_color}  (RGB?????)')
        # Convert the image to the HSV color space
        hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        # Convert the average color to the HSV color space
        average_color_hsv = cv.cvtColor(
            np.uint8([[average_color]]), cv.COLOR_BGR2HSV)[0][0]

        # count the objects
        count_objects(image, average_color_hsv)

        # exit the program
        break

    if cv.waitKey(1) & 0xFF == ord('q'):
        break


cv.waitKey(0)
cv.destroyAllWindows()