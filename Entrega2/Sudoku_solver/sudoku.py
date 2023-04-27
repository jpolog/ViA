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
    for i in range(len(rs)):
        if len(rs[i]) == n:
            quad.append(rs[i])
        else:
            other.append(cs[i])
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

        # Buscamos los contornos interiores dentro del cuadrado de mayor área. Debe haber 81
        # numpy arrays de contornos interiores y números dentro de los contornos interiores
        inside = np.empty(9, dtype=list)
        numbers_inside = np.full(9,None, dtype=object)


        j = 0   # index of inside matrix
        for c in range(len(quad_interior)):
            in_points = 0
            ### Cambiar esto por un for-else
            for i in range(len(quad_interior[c])):
                # comprueba si el punto está completamente dentro del cuadrado de mayor área
                if cv.pointPolygonTest(quad_borders[maxC], quad_interior[c][i].tolist(), False) == 1:
                    in_points += 1
            # si todos los puntos están dentro del cuadrado de mayor área
            # se añade a la lista de contornos interiores
            if in_points == len(quad_interior[c]):
                inside[j] = quad_interior[c]            
                # para cada contorno de numeros, 
                # se comprueba si está dentro del contorno
                for n in contour_numbers:
                    if cv.pointPolygonTest(quad_interior[c], n[0].tolist(), False) == 1:
                        # si está dentro, se añade a la lista de números de dentro del cuadrado
                        numbers_inside[j]= n
                        break
                # if not, a None is added
                else:
                    pass
                
                j = j+1
                if j==9:    # máximo de contornos interiores del sudoku
                    break

        # inside tiene que tener 81 contornos
        if len(inside) != 9:
            return None, None, None
        

        # crea una ventana para mostrar los contornos de inside y su índice i dentro de inside
        cv.namedWindow('inside', cv.WINDOW_NORMAL)
        cv.resizeWindow('inside', 600, 600)
        image = np.zeros((600,600,3), np.uint8)
        for i in range(len(inside)):
            cv.drawContours(image, [inside[i]], -1, (0, 255, 0), 2)
            cv.putText(image, str(i), tuple(inside[i][0]), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv.imshow('inside', image)

        # reordena inside por la posición de sus esquinas en el eje y
        # reordena numbers_inside por el mismo orden que inside
        # Ambas matrices tiene las mismas dimensiones
        idx = np.argsort([polygon[0][1] for polygon in inside])
        inside = inside[idx]
        numbers_inside = numbers_inside[idx]
        # y luego de 3 en 3 por la posición de sus esquinas en el eje x
        for i in range(0,len(inside),3):
            idx = np.argsort([polygon[0][0] for polygon in inside[i:i+3]])
            inside[i:i+3] = inside[i:i+3][idx]
            numbers_inside[i:i+3] = numbers_inside[i:i+3][idx]



        
        # retornamos el índice del contorno de borde más grande
        #  y los contornos interiores
        return maxC, inside, numbers_inside
        
        
#######################################################################
############  MODELOS DE RECONOCIMIENTO DE NÚMEROS  ###################
#######################################################################

# basándonos en imágenes de los números de 0 a 9
# creamos un modelo de reconocimiento de números
# a partir de los contornos de los números
def createModels():
    # creamos un diccionario con los contornos de los números
    models = []
    for i in range(9):
        img = cv.imread('numbers/{}.png'.format(i+1))
        contours = extractContours(img)
        models.append(contours[0])
    return models

# creamos los modelos de los números
numberModels = createModels()

# Definimos la función que calcula el invariante
# de forma basado en las frecuencias dominantes
# esta función recibe un contorno y produce descriptor de tamaño wmax*2+1
def invar(c, wmax=10):
    x,y = c.T
    z = x+y*1j
    f  = fft(z)
    fa = abs(f)

    s = fa[1] + fa[-1]

    v = np.zeros(2*wmax+1)
    v[:wmax] = fa[2:wmax+2]
    v[wmax:] = fa[-wmax-1:]

   
    if fa[-1] > fa[1]:
        v[:-1] = v[-2::-1]
        v[-1] = fa[1]

    return v / s

# comparamos los invariantes de los contornos 
# encontrados en la imagen con el modelo y señalamos
# los que son muy parecidos
MAXDIST = 0.05

# como el invariante del 6 y del 9 son muy parecidos
# (asumimimos que el 6 es un 9 rotado)
# definimos una función que comprueba si el contorno
# es un 6 o un 9, calculando la distribución de píxeles
# en el contorno
def is_6_or_9(contour):
    # centroide del contorno
    M = cv.moments(contour)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    # dibujamos el contorno y el centroide en una ventana (en el centro)
    window = np.zeros((1500,1500,3), np.uint8)
    # el contorno debe estar en el centro de la ventana
    cv.drawContours(window, [contour], -1, (0, 255, 0), 2)
    cv.circle(window, (cx, cy), 5, (0, 0, 255), -1)
    cv.imshow('window', window)
    

    # si es un 6, el centroide estará en la parte superior
    # si es un 9, el centroide estará en la parte inferior
    # calculamos la distribución de píxeles en el contorno
    x, y, w, h = cv.boundingRect(contour)
    center_y = y + h / 2
    if cy > center_y:
        return 6
    else:
        return 9


def identifyNumbers(contours, models):
    #almacenamos los números encontrados en su posición del sudoku
    # array de 9 posiciones a 0
    numbers = [0]*9
    lenNumbers = 0
    j = 0
    modelwindow = np.zeros((500,500,3), np.uint8)

    for m in range(len(models)):
        # show the model
        modelwindow = np.zeros((500,500,3), np.uint8)
        cv.drawContours(modelwindow, [models[m]], -1, (0,255,0), 1)
        cv.imshow('model', modelwindow)
        j+=1
        
        invmodel = invar(models[m])
        for i in range(len(contours)):
            c = contours[i]
            if c is None:
                continue
            if c is not None:
                if np.linalg.norm(invar(c)-invmodel) < MAXDIST:
                    if m == 5 or m == 8:
                        number = is_6_or_9(contours[i]) -1
                        if m != number: # ha detectado un 6 como 9 o viceversa
                            continue

                    # se sustituye el contorno por el número
                    print('number {} found'.format(m+1))
                    numbers[i] = m+1
                    lenNumbers += 1
                    break # solo 1 número por contorno
        
            

    # si todos los números han sido identificados
    # se devuelve la lista de números
    lenContours = sum(1 for i in contours if i is not None)
    if lenNumbers == lenContours:
        # se devuelve la lista de números
        # en un array de 3x3
        return np.array(numbers).reshape(3,3)
    else:
        return None


########################################################################
############  RESOLUCIÓN DEL SUDOKU  ###################################
########################################################################





found=False # hemos encontrado el sudoku?
print('\nstarting program\n')
for (key,frame) in autoStream():


    # cambiamos el modo de visualización
    if key == ord('c'):
        shcont = not shcont

    contours = extractContours(frame)

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
        maxC, inside, numbers_inside = findSudokuContours(quad_borders, quad_interior, contour_numbers)

        # mientras no encontremos un contorno con 4 esquinas y 81 contornos interiores
        # seguimos buscando
        if maxC != None and len(inside) == 9:
            # Se ha encontrado un sudoku
            found = True

    
    if found:
        sudoku = identifyNumbers(numbers_inside, numberModels)
        if sudoku is not None:
            print("\n\n\nSUDOKU ENCONTRADO:")
            print(sudoku)
            print("\n\n\n")

            
    



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


