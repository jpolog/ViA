import numpy as np
import cv2   as cv
from collections import deque


print("b) Calibración aproximada\n")

# Puntos para medir el objeto en la imagen
points = deque(maxlen=2)
state = 0

# Imagen objeto de referencia
img = cv.imread("./lamp.jpg", cv.IMREAD_GRAYSCALE)
image_h, image_w = img.shape[:2]

# primero se mide el objeto verticalmente y luego horizontalmente
# cuando se marcan los puntos se cambia el estado y se calculan los valores
def fun(event, x, y, flags, param):
    global image_object_h, image_object_w, state
    if event == cv.EVENT_LBUTTONDOWN:
        points.append((x,y))
        cv.circle(img, (x,y), 5, (0,0,255), -1)

        if len(points) == 2 and state == 0:
            # Medidas de la imagen del objeto
            #> ESTO SE BASA EN medidor.py
            image_object_h = np.linalg.norm(np.array(points[1])-points[0])  # pixels
            state = 1
            points.clear()
        elif len(points) == 2 and state == 1:
            # Medidas de la imagen del objeto
            image_object_w = np.linalg.norm(np.array(points[1])-points[0]) # pixels
            state = 2   

cv.namedWindow("image")
cv.setMouseCallback("image", fun)

while True:
    cv.imshow("image", img)
    if state == 2:  # all points marked
        break
    if cv.waitKey(1) & 0xFF == ord("q"):
        break



# marcar puntos en la imagen y medir la distancia en píxeles
#> Medidas reales del objeto ESTO SE PUEDE QUEDAR ASÍ O PEDIRLAS AL USUARIO
object_h = 168  # cm
object_w = 27  # cm

# Distancia entre la cámara y el objeto
#> ESTO SE PUEDE QUEDAR ASÍ O PEDIRLAS AL USUARIO
distance = 150  # cm

# Calculamos la focal length
focal_length_a = (image_object_h*distance)/object_h

# Calculamos el FOV vertical
vFov_a = 2*np.arctan(image_h/(2*focal_length_a))

# Calculamos FOV horizontal
hFov_a = 2*np.arctan(image_w/(2*focal_length_a))

# print fov en grados
print(f"Focal length: {focal_length_a} pix")
print("FOV vertical: {}º".format(vFov_a * 180 / np.pi))
print("FOV horizontal: {}º".format(hFov_a * 180 / np.pi))



