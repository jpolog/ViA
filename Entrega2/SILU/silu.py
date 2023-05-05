#!/usr/bin/env python

import os
import cv2          as cv
from umucv.stream import autoStream
import numpy as np
from numpy.fft import fft

IMG_DIR_PATH = './models'

#######################################################################
####################  EXTRAER LOS CONTORNOS  ##########################
#######################################################################

def binarize(gray):
    _, r = cv.threshold(gray, 128, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    return r
    
def extractContours(image):
    g = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    if black:
        g = 255-g
    b = binarize(g)  
    cv.imshow('binarize', b)
    contours, _ = cv.findContours(b.copy(), cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)
    contours = [ c.reshape(-1,2) for c in contours ]
    contours = sorted(contours, key=cv.contourArea, reverse=True)
    return contours


# Varía en función de las diemensiones de la imagen con la que se trabaje
def razonable(c, image):
    return (image.shape[0]*image.shape[1])*0.95 >= cv.contourArea(c) >= 0.0004*(image.shape[0]*image.shape[1])


def orientation(x):
    return cv.contourArea(x.astype(np.float32),oriented=True) >= 0

# para elegir si mostramos la imagen original o todos los contornos
black = True
shcont = True

def redu(c, eps=0.5):
    red = cv.approxPolyDP(c,eps,True)
    return red.reshape(-1,2)

def polygons(cs,prec=2):
    rs = [ redu(c,prec) for c in cs ]
    return rs


#######################################################################
############  MODELOS DE RECONOCIMIENTO DE NÚMEROS  ###################
#######################################################################

# basándonos en imágenes de los números y letras de las matrículas
# creamos un modelo de reconocimiento de números
# a partir de los contornos de los números
def createModels():
    alphaModels = [] # letras del alfabeto
    numberModels = [] # números 0-9
    files = os.listdir(IMG_DIR_PATH)
    # ordenamos alfabéticamente
    files.sort()
    print("files: ",files)
    for filename in files:
        if filename.endswith(".png"):
            img = cv.imread(os.path.join(IMG_DIR_PATH, filename))
            contours = extractContours(img)
            name = filename.split('.')[0]
            if name.isalpha():
                alphaModels.append(contours[0])
            else:
                numberModels.append(contours[0])

    return alphaModels, numberModels
    
   
# creamos los modelos de los números y letras
alphaModels, numberModels = createModels()

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

# Diferentes umbrales para el reconocimiento de los números
# y las letras
MAXDIST_NUM = 0.04
MAXDIST_ALPHA = 0.1

# como el invariante del 6 y del 9 son muy parecidos
# (asumimimos que el 6 es un 9 rotado)
# definimos una función que comprueba si el contorno
# es un 6 o un 9, calculando la distribución de píxeles
# en el contorno
def is_6_or_9(contour):
    # centroide del contorno
    M = cv.moments(contour)
    cy = int(M['m01']/M['m00'])
    
    # si es un 6, el centroide estará en la parte superior
    # si es un 9, el centroide estará en la parte inferior
    # calculamos la distribución de píxeles en el contorno
    _, y, _, h = cv.boundingRect(contour)
    center_y = y + h / 2
    if cy > center_y:
        return 6
    else:
        return 9
    
# retorna la letra a partir de su posición en 
# la lista de letras válidas de las matrículas
def getAlpha(position):
    dict = {0:'B', 1:'C', 2:'D', 3:'F', 4:'G', 5:'H',
            6:'J', 7:'K', 8:'L', 9:'M', 10:'N', 11:'P',
            12:'R', 13:'S', 14:'T', 15:'V', 16:'W', 17:'X',
            18:'Y', 19:'Z'}
    return dict[position]


def identifySymbols(contours,alphaModels,numberModels):
    #almacenamos los símbolos encontrados en su posición 
    # dentro de la matrícula
    numbers = []
    alpha = []

    # detectar si hay dos filas de contornos 
    # con el formato de las matrículas en españa puede haber
    # 2 filas de contornos (números arriba y letras abajo) o 1 fila (números y después letras)
    # si algún contorno está debajo del final de otro es que hay 2 filas
    if any(cv.boundingRect(contours[i])[1] > cv.boundingRect(contours[i+1])[1] + cv.boundingRect(contours[i+1])[3] for i in range(len(contours)-1)):
        # ordenamos verticalmente
        contours = sorted(contours, key=lambda x: cv.boundingRect(x)[1])
        # los 4 primeros contornos (arriba) son los números
        # se ordenan de izquierda a derecha
        numbers = sorted(contours[:4], key=lambda x: cv.boundingRect(x)[0])
        # los 3 últimos contornos (abajo) son las letras
        # se ordenan de izquierda a derecha
        alpha = sorted(contours[4:], key=lambda x: cv.boundingRect(x)[0])
    else:
        # se ordenan de izquierda a derecha
        contours = sorted(contours, key=lambda x: cv.boundingRect(x)[0])

    
    
    # identificar los números
    invmodel = [0]*10
    for i in range(len(numberModels)):
        invmodel[i] = invar(numberModels[i])
        
    for c in contours[:4]:
        # show contour
        cv.imshow('contour actual', cv.drawContours(img.copy(), [c], -1, (0,255,0), 1))
        for m in range(len(numberModels)):
            invmodel = invar(numberModels[m])
            cv.imshow('model', cv.drawContours(img.copy(), [numberModels[m]], -1, (0,255,0), 1))

            if np.linalg.norm(invar(c)-invmodel) < MAXDIST_NUM:
                if m == 5 or m == 8:
                    number = is_6_or_9(c) -1
                    if m != number: # ha detectado un 6 como 9 o viceversa
                        continue
                # se registra el número
                numbers.append(m)
                cv.waitKey(0)

    invmodel = [0]*20
    # identificar las letras
    for i in range(len(alphaModels)):
        invmodel[i] = invar(alphaModels[i])

    for c in contours[4:]:
        for m in range(len(alphaModels)):
            invmodel = invar(alphaModels[m])

            if np.linalg.norm(invar(c)-invmodel) < MAXDIST_ALPHA:
                letter = getAlpha(m)
                # se registra la letra
                alpha.append(letter)
        
    if len(numbers) != 4 or len(alpha) != 3:
        print('No se ha identificado una matrícula válida')
        return None
    else:
        return numbers + alpha
    
contours = []
borders = []
border = None
## Programa principal
for (key,frame) in autoStream():
    #1. abrir la imagen
    img = frame.copy()
    #2. extraer los contornos
    contours = extractContours(frame)
    # seleccionamos contornos OSCUROS de tamaño razonable
    contours = [c for c in contours if razonable(c,img) and not orientation(c) ]
    borders = [c for c in contours if razonable(c,img) and orientation(c) ]
    if len(borders) != 0:
        borders = polygons(border)

        for b in borders:
            if len(borders) == 4:
                border = b
    
    if border is not None:
        #detectamos los contornos dentro del borde
        contours = [c for c in contours if cv.pointPolygonTest(border, tuple(c[0]), False) >= 0]

    
        

    cv.imshow('contours', cv.drawContours(img.copy(), contours, -1, (0,255,0), 1))
    #show contours
    #cv.waitKey(0)
    #3. identificar los símbolos
    symbols = identifySymbols(contours,alphaModels,numberModels)

    if symbols is not None:
        print("Matrícula: ", symbols)
