import numpy as np
import cv2   as cv

print("c) Altura mínima de la cámara\n")

# Dimensiones de la pista
# Tomaremos el lado más largo como altura (depende de cómo esté orientada la cámara en la calibración anterior)
pista_x = 28*100  # cm
pista_y = 15*100  # cm

# cargamos los valores de la calibración PRECISA?? APROXIMADA??? del ejercicio anterior
#> Esto se podria hacer con un fichero del codigo anterior, de momento los meto a mano
focal_length_a =  535.8
# En radianes !!!!!
vFov_a =  1.0767
hFov_a =  0.8422


# Calculamos altura mínima para que la cámara vea toda la pista en horizontal
#> HAY UN POCO DE LÍO CON LAS H Y LAS V SORRY LA NOMENCLATURA ES UNA MIERDA :(
min_height_v = (pista_x/2) / np.tan(vFov_a/2)
print(min_height_v)
# En vertical
min_height_h = (pista_y/2) / np.tan(hFov_a/2)
print(min_height_h)

# altura mínima será la mayor de las dos
min_height = max(min_height_h, min_height_v)

print(f"Altura mínima: {min_height/100} m\n")