import cv2 as cv
import numpy as np
from umucv.htrans import htrans
from umucv.stream import autoStream
from collections import deque

# distancia entre dos puntos en el plano
def dist(p1,p2):
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

# Lee las coordenadas (x,y) de los puntos de referencia
# de un fichero de texto y la distancia real entre
# los dos primeros puntos (en la unidad correspondiente)
# el formato será:
# d \n x1,y1 \n x2,y2 \n ... \n xn,yn \n
def readRef(filename):
    f = open(filename,'r')
    d = float(f.readline())
    refPts = []
    for line in f:
        refPts.append([int(x) for x in line.split(',')])
    return d, refPts


# Detecta el evento de click del ratón
# y almacena las coordenadas de los puntos
# se detectan tantos clicks como puntos se han leído
def mouse_callback(event, x, y, flags, numPts):
    if event == cv.EVENT_LBUTTONDOWN:
        global imgPts
        global img
        if len(imgPts) < numPts:
            print('Point %d: (%d,%d) registered' % (len(imgPts),x,y))
            imgPts.append([x,y])
            cv.circle(img,(x,y),5,(0,0,255),-1)

# calcula la distancia real entre dos puntos de referencia
# a partir de la distancia entre los dos primeros puntos
# y la distancia entre los dos primeros puntos en la imagen
#   · d: distancia real entre los dos primeros puntos
#   · mPtsReal: puntos marcados en el mundo real
#   · refPtsReal: puntos de referencia en el mundo real
def calculateRealDist(d, mPtsReal, refPtsReal):
    refDistReal = dist(refPtsReal[0],refPtsReal[1])
    mDistReal = dist(mPtsReal[0],mPtsReal[1])
    print('Distancia real (pix) entre puntos de referencia: %f' % refDistReal)
    print('Distancia (pix) entre puntos de referencia en la imagen: %f' % mDistReal)
    print('d: %f' % d)

    return d * mDistReal / refDistReal
    


# calcula la matriz de transformación que rectifica la imagen
# a partir de los puntos de referencia y sus coordenadas reales
def calculateHomography(imgPts, refPts):
    H,_ = cv.findHomography(imgPts, refPts)
    return H

# rectifica la imagen
def rectify(img, H):
    return cv.warpPerspective(img, H, (img.shape[1],img.shape[0]))


########################################################################
############ HERRAMIENTAS DE MEDICIÓN ##################################
########################################################################
def medir(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        mPtsImg.append((x,y))
        print('Real Point %d: (%d,%d) registered' % (len(mPtsImg),x,y))


########################################################################
############  PROGRAMA PRINCIPAL  ######################################
########################################################################

print('\nstarting program\n')
for (key,frame) in autoStream():
    img = frame.copy()
    cv.imshow('original',img)

    # lee los puntos de referencia
    d, refPts = readRef('ref.txt')
    
    # imgPts es np array de puntos de la imagen
    # que se van rellenando con los clicks del ratón
    imgPts = []


    # espera a que se marquen los puntos de referencia
    cv.setMouseCallback('original', mouse_callback, len(refPts))
    while len(imgPts) < len(refPts):
        cv.waitKey(1)
        # muestra la imagen
        cv.imshow('original',img)
        continue

    # matrices de puntos en np array
    imgPts = np.array(imgPts)
    refPts = np.array(refPts)
    # calcula la matriz de transformación
    H = calculateHomography(imgPts, refPts)

    # y la inversa
    IH = np.linalg.inv(H)

    # rectifica la imagen
    imgRect = rectify(img, H)
    # muestra la imagen rectificada
    cv.imshow('rectified',imgRect)

    # medición de distancias
    mPtsImg = deque(maxlen=2)
    cv.setMouseCallback('original', medir)
    while len(mPtsImg) < 2:
        for p in mPtsImg:
            cv.circle(img, p,3,(255,0,255),-1)
        cv.imshow('original',img)
        cv.waitKey(1)
        continue

    # se dibuja la línea en la imagen original
    cv.line(img, mPtsImg[0],mPtsImg[1],(0,0,255))
    # se dibuja la línea en el mundo real
    mPtsReal = htrans(H, np.array(mPtsImg)).astype(int)
    cv.circle(imgRect, tuple(mPtsReal[0]),3,(255,0,255),-1)
    cv.circle(imgRect, tuple(mPtsReal[1]),3,(255,0,255),-1)
    cv.line(imgRect, mPtsReal[0],mPtsReal[1],(0,0,255))


    # calcula la distancia real entre los puntos
    dReal = calculateRealDist(d, mPtsReal, refPts)
    print('Real distance: %f' % dReal)

    
    # muestra la imagen original y la rectificada
    cv.imshow('original',img)
    cv.imshow('rectified',imgRect)

    # espera a que se pulse una tecla
    cv.waitKey(0)
    cv.destroyAllWindows()








    


        
    
