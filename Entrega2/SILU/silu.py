#!/usr/bin/env python

import os
import cv2          as cv
from umucv.stream import autoStream
import numpy as np
from numpy.fft import fft

# if last arg is advanced ---> activate advanced mode
advanced = len(os.sys.argv) > 1 and os.sys.argv[-1] == '--advanced'


IMG_DIR_PATH = './models'

#######################################################################
####################  EXTRAER LOS CONTORNOS  ##########################
#######################################################################

def binarize_white(gray):
    _, r = cv.threshold(gray, 200, 255, cv.THRESH_BINARY)
    return r

def binarize_black(gray):
    _, r = cv.threshold(gray, 70, 255, cv.THRESH_BINARY)
    return r

SIGMA = [0.03]
cv.namedWindow('matricula')
if advanced:
    cv.createTrackbar('Blur', 'matricula', int(SIGMA[0]*100), 500, lambda v: SIGMA.insert(0,v/100))
def extractContours(image, black=True):
    g = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    #gauss
    g = cv.GaussianBlur(g, (0,0), SIGMA[0])

    cv.imshow('g_b', g)
     

    if black:
        b = 255-g
        b = binarize_black(b)
        cv.imshow('b_b', b)
    else:
        b = binarize_white(g)
    contours, _ = cv.findContours(b.copy(), cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)
    contours = [ c.reshape(-1,2) for c in contours ]
    contours = sorted(contours, key=cv.contourArea, reverse=True)
    return contours




# Varía en función de las diemensiones de la imagen con la que se trabaje
def razonable(c, image):
    return (image.shape[0]*image.shape[1])*0.95 >= cv.contourArea(c) >= 0.008*(image.shape[0]*image.shape[1])


def orientation(x):
    return cv.contourArea(x.astype(np.float32),oriented=True) >= 0

# para elegir si mostramos la imagen original o todos los contornos
shcont = True

def redu(c, eps=0.5):
    red = cv.approxPolyDP(c,eps,True)
    return red.reshape(-1,2)

def polygons(cs,n,prec=2):
    rs = [ redu(c,prec) for c in cs ]
    sq = []
    for r in rs:
        if len(r) == n:
            sq.append(r)
    return sq


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
MAXDIST_NUM = [0.08]
MAXDIST_ALPHA = [0.1]
#trackbar para cada valor
if advanced:
    cv.createTrackbar('Umbral Números', 'matricula', int(MAXDIST_NUM[0]*1000), 400, lambda v: MAXDIST_NUM.insert(0,v/1000))
    cv.createTrackbar('Umbral Letras', 'matricula', int(MAXDIST_ALPHA[0]*100), 50, lambda v: MAXDIST_ALPHA.insert(0,v/100))


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
        contours[:4] = sorted(contours[:4], key=lambda x: cv.boundingRect(x)[0])
        # los 3 últimos contornos (abajo) son las letras
        # se ordenan de izquierda a derecha
        contours[4:] = sorted(contours[4:], key=lambda x: cv.boundingRect(x)[0])
    else:
        # se ordenan de izquierda a derecha
        contours = sorted(contours, key=lambda x: cv.boundingRect(x)[0])

    
    
    # identificar los números
    invmodel = [0]*10
    for i in range(len(numberModels)):
        invmodel[i] = invar(numberModels[i])
    
    for i in range(len(contours[:4])):
        c = contours[i]
        # show contour
        for m in range(len(numberModels)):
            invmodel = invar(numberModels[m])
            #cv.waitKey(0)
            diff = np.linalg.norm(invar(c)-invmodel)
            if  diff < MAXDIST_NUM[0]:
                if m == 6 or m == 9:
                    number = is_6_or_9(c)
                    if m != number: # ha detectado un 6 como 9 o viceversa
                        continue
                elif m == 0: # evitar que se detecte un 8 como un 0 (como el 0 va primero, ya no sigue buscando)
                    invmodel = invar(numberModels[8])
                    newDiff = np.linalg.norm(invar(c)-invmodel)
                    if newDiff < diff:
                        numbers.append(8)
                        break
                    else:
                        numbers.append(m)
                        break
                else:
                    # se registra el número
                    numbers.append(m)
                    break

    invmodel = [0]*20
    # identificar las letras
    for i in range(len(alphaModels)):
        invmodel[i] = invar(alphaModels[i])

    for c in contours[4:]:
        for m in range(len(alphaModels)):
            invmodel = invar(alphaModels[m])

            if np.linalg.norm(invar(c)-invmodel) < MAXDIST_ALPHA[0]:
                letter = getAlpha(m)
                # algunos casos requieren más precisión
                if letter == 'J' or letter == 'L':
                    if np.linalg.norm(invar(c)-invmodel) < 0.025:
                        alpha.append(letter)
                        break
                    else:
                        continue
                # se registra la letra
                alpha.append(letter)
                break
        
    if len(numbers) != 4 or len(alpha) != 3:
        print('No se ha identificado una matrícula válida')
        return None
    else:
        return numbers + alpha
    


cv.namedWindow("matricula")


contours = []
borders = []
border = None
found = False
plate = []
## Programa principal
for (key,frame) in autoStream():
    #1. abrir la imagen
    img = frame.copy()
    #2. extraer los contornos
    borders = extractContours(frame, black=False)
    # seleccionamos contornos BLANCOS de tamaño razonable 
    borders = [c for c in borders if razonable(c,img) and not orientation(c) ]
    # precisión alta para que se quede solamente el recuadro blanco
    borders = polygons(borders,4,35)
    if len(borders) == 0:
        continue
    # nos quedamos con el más pequeño
    border = borders[-1]
    #recortamos la imagen al tamaño del borde (ROI)
    x,y,w,h = cv.boundingRect(border)
    img = img[y:y+h,x:x+w]
    contours = extractContours(img)
    # seleccionamos contornos OSCUROS de tamaño razonable
    contours = [c for c in contours if razonable(c,img) and not orientation(c) ]
    # si se ha detectado el rectángulo con la letra del país
    # se elimina de los contornos
    if len(contours) > 7:
        contours = contours[1:]
    b = np.zeros_like(img, dtype=np.uint8)
    cv.imshow('contours', cv.drawContours(b, contours, -1, (0,255,0), 1))
    #3. identificar los símbolos
    symbols = identifySymbols(contours,alphaModels,numberModels)

    if symbols is not None:
        if symbols != plate:
            found = False
        if not found:
            print("Matrícula: ", symbols)
            plate = symbols
            found = True

    cv.imshow('matricula', cv.drawContours(img.copy(), contours, -1, (0,255,0), 1))