import numpy as np
import cv2   as cv

print("c) Altura mínima de la cámara\n")

# Dimensiones de la pista
# Tomaremos el lado más largo como altura (depende de cómo esté orientada la cámara en la calibración anterior)
pista_y = 28*100  # cm
pista_x = 15*100  # cm

# cargamos los valores de la calibración PRECISAA del ejercicio anterior

# En radianes 
# read value from file vFOV.txt
f = open('vFOV.txt','r')
vFov_a = float(f.read())
f = open('hFOV.txt','r')
hFov_a = float(f.read())


# Calculamos altura mínima para que la cámara vea toda la pista en horizontal
min_height_v = (pista_x/2) / np.tan(vFov_a/2)
print(f'Altura mínima en vertical: {min_height_v/100} m')
# En vertical
min_height_h = (pista_y/2) / np.tan(hFov_a/2)
print(f'Altura mínima en horizontal: {min_height_h/100} m')

# altura mínima será la mayor de las dos
min_height = max(min_height_h, min_height_v)

print(f"Altura mínima: {min_height/100} m\n")