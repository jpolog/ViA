#!/usr/bin/env python

import cv2          as cv
from umucv.stream import autoStream
import numpy as np
# la fft de numpy
from numpy.fft import fft

def binarize(gray):
    _, r = cv.threshold(gray, 128, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    return r
    
def extractContours(image):
    g = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    if black:
        g = 255-g
    b = binarize(g)  
    contours, _ = cv.findContours(b.copy(), cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)
    contours = [ c.reshape(-1,2) for c in contours ]
    contours = sorted(contours, key=cv.contourArea, reverse=True)
    return contours


###> AJUSTAR ESTOS PARÁMETROS PARA QUE FUNCIONE BIEN
def razonable(c):
    return 1000**2 >= cv.contourArea(c) >= 10**2


def orientation(x):
    return cv.contourArea(x.astype(np.float32),oriented=True) >= 0

black = True

# para elegir si mostramos la imagen original o todos los contornos
shcont = True

#######################################################################
def redu(c, eps=0.5):
    red = cv.approxPolyDP(c,eps,True)
    return red.reshape(-1,2)

def polygons(cs,n,prec=2):
    rs = [ redu(c,prec) for c in cs ]
    quad = []
    other = []
    for r in rs:
        if len(r) == n:
            quad.append(r)
        else:
            other.append(r)
    return quad, other

def findSudokuContours(quad_borders, quad_interior, contour_numbers):
    # Recorremos los contornos de borde, y nos quedamos 
        # con el contorno de mayor área
        maxArea = 0
        maxC = None
        currentC = -1   #count the number of contours
        for c in quad_borders:
            currentC += 1
            area = cv.contourArea(c)
            if area > maxArea:
                maxArea = area
                maxC = currentC
                # Se almacenan las 4 esquinas del cuadrado de mayor área
                corners = []
                for point in quad_borders[maxC]:
                    corners.append(point.tolist())  

        # Buscamos los contornos interiores dentro del cuadrado de mayor área
        inside = []
        numbers_inside = []

        for c in quad_interior:
            in_points = 0
            ### Cambiar esto por un for-else
            for i in range(len(c)):
                # comprueba si el punto está completamente dentro del cuadrado de mayor área
                if cv.pointPolygonTest(quad_borders[maxC], c[i].tolist(), False) == 1:
                    in_points += 1
            # si todos los puntos están dentro del cuadrado de mayor área
            # se añade a la lista de contornos interiores
            if in_points == len(c):
                inside.append(c)

                # para cada contorno de numeros, 
                # se comprueba si está dentro del contorno
                for n in contour_numbers:
                    if cv.pointPolygonTest(c, n[0].tolist(), False) == 1:
                        # si está dentro, se añade a la lista de números de dentro del cuadrado
                        numbers_inside.append(n)
                        break
                # if not, a None is added
                else:
                    numbers_inside.append(None)


        # retornamos el índice del contorno de borde más grande
        #  y los contornos interiores
        return maxC, inside
        
        
#######################################################################
############  MODELOS DE RECONOCIMIENTO DE NÚMEROS  ###################
#######################################################################

# basándonos en imágenes de los números de 0 a 9
# creamos un modelo de reconocimiento de números
# a partir de los contornos de los números
def createModels():
    # creamos un diccionario con los contornos de los números
    models = {}
    for i in range(9):
        img = cv.imread('numbers/{}.png'.format(i+1))
        print('numbers/{}.png'.format(i+1))
        contours = extractContours(img)
        models[i] = contours[0]
    return models

# creamos los modelos de los números
numberModels = createModels()

# Definimos la función que calcula el invariante
# de forma basado en las frecuencias dominantes
# esta función recibe un contorno y produce descriptor de tamaño wmax*2+1
def invar(c, wmax=10):
    x,y = c.T                       # separamos las coordenadas x e y
    z = x+y*1j                      # convertimos los puntos en números complejos
    f  = fft(z)                     # calculamos la transformada de Fourier discreta
    fa = abs(f)                     # tomamos el módulo para conseguir invarianza a rotación
                                    # y punto de partida
    s = fa[1] + fa[-1]              # La amplitud de la frecuencia 1 nos da el tamaño global de la figura
                                    # y servirá para normalizar la escala
    v = np.zeros(2*wmax+1)          # preparamos espacio para el resultado
    v[:wmax] = fa[2:wmax+2];        # cogemos las componentes de baja frecuencia, positivas
    v[wmax:] = fa[-wmax-1:];        # y las negativas.
                                    # Añadimos también la frecuencia -1, que tiene
                                    # que ver con la "redondez" global de la figura
   
    if fa[-1] > fa[1]:              # normalizamos el sentido de recorrido
        v[:-1] = v[-2::-1]          # (El círculo dominante debe moverse en sentido positivo)
        v[-1] = fa[1]
    
    return v / s                    # normalizamos el tamaño

# comparamos los invariantes de los contornos 
# encontrados en la imagen con el modelo y señalamos
# los que son muy parecidos
MAXDIST = 0.15


def identifyNumbers(contours, models):
    numbers = [] #almacenamos los números encontrados en su posición del sudoku
    lenContours = 0
    lenNumbers = 0
    for m in models:
        invmodel = invar(m)
        for i in range(len(contours)):
            c = contours[i]
            if c is not None:
                lenContours += 1
                if np.linalg.norm(invar(c)-invmodel) < MAXDIST:
                    # se sustituye el contorno por el número
                    numbers[i] = m
                    lenNumbers += 1
                    break # solo 1 número por contorno

    # si todos los números han sido identificados
    # se devuelve la lista de números
    if lenNumbers == lenContours:
        return numbers
    
    else: return None


#######################################################################



found=False # hemos encontrado el sudoku?
print('\nstarting program\n')
for (key,frame) in autoStream():


    # cambiamos el modo de visualización
    if key == ord('c'):
        shcont = not shcont

    g = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    if black:
        g = 255-g
    
    b = binarize(g)
    contours = extractContours(b)

    # seleccionamos contornos OSCUROS de tamaño razonable
    contours_borders = [c for c in contours if razonable(c) and not orientation(c) ]
    # seleccionamos contornos CLAROS de tamaño razonable
    contours_interior = [c for c in contours if razonable(c) and orientation(c) ]


    ##> Ajustar precisión para que funcione bien
    # obtenemos los contornos de los cuadrados
    # y se separan los que no son cuadrados (números en el caso de contornos de borde)
    quad_borders, contour_numbers = polygons(contours_borders,n=4,prec=3)
    quad_interior, _ = polygons(contours_interior,n=4,prec=3)


    # Si no hemos encontrado el sudoku, buscamos entre los contornos
    # Obtendremos el índice del contorno de mayor área
    # Y los contornos interiores que estén dentro de él
    if not found:
        maxC, inside = findSudokuContours(quad_borders, quad_interior)

        # mientras no encontremos un contorno con 4 esquinas y 81 contornos interiores
        # seguimos buscando
        if maxC != None and len(inside) == 9:
            # Se ha encontrado un sudoku
            found = True

    
    if found:
        sudoku = identifyNumbers(contour_numbers, numberModels)
        if sudoku != None:
            print(sudoku)
            
    



    if shcont:
        # en este modo de visualización mostramos en colores distintos
        # las manchas oscuras y las claras
        result = np.zeros_like(frame)
        cp = [c for c in contours if orientation(c) ]
        cn = [c for c in contours if not orientation(c) ]
        
        cv.drawContours(result, cp, contourIdx=-1, color=(255,128,128), thickness=1, lineType=cv.LINE_AA)
        cv.drawContours(result, cn, -1, (128,128,255), 1)
           
    else:
        # en este modo de visualización mostramos los contornos que pasan el primer filtro
        result = frame
        #cv.drawContours(result, quad_borders, -1, (255,0,0), cv.FILLED)
    
        # En el cuadrado de mayor área dibujamos los puntos de las esquinas
        if found:
            for point in quad_borders[maxC]:
                cv.circle(frame, (point[0], point[1]), 5, (255,0,0), -1)

            # En los contornos interiores dibujamos los puntos de las esquinas
            for c in inside:
                for point in c:
                    cv.circle(frame, (point[0], point[1]), 5, (0,255,0), -1)
    cv.imshow('shape recognition',result)

cv.destroyAllWindows()


