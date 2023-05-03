import cv2 as cv
import numpy as np
from umucv.htrans import htrans
from umucv.stream import autoStream
from collections import deque

numPts = 4
numPtsReg= 0
# puntos de cada cuadrado
# las esquinas se deben marcar en el mismo orden en los dos cuadrados!!!!
ptsC1 = []
ptsC2 = []


# Detecta el evento de click del ratón
# y almacena las coordenadas de los puntos
# se detectan tantos clicks como puntos se han leído
def mouse_callback(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        global numPts, ptsC1, ptsC2, numPtsReg
        if numPtsReg < numPts:
            print('Point %d: (%d,%d) registered in C1' % (numPtsReg,x,y))
            ptsC1.append([x,y])
            cv.circle(img,(x,y),5,(0,0,255),-1)
            numPtsReg += 1
        elif numPtsReg >= numPts and numPtsReg < 2*numPts:
            print('Point %d: (%d,%d) registered in C2' % (numPtsReg,x,y))
            ptsC2.append([x,y])
            cv.circle(img,(x,y),5,(255,0,0),-1)
            numPtsReg += 1
        else:
            print('All points registered')
            

# calcula la matriz de transformación que rectifica la imagen
# a partir de los puntos de referencia y sus coordenadas reales
def calculateHomography(imgPts, refPts):
    H,_ = cv.findHomography(imgPts, refPts)
    return H

# rectifica la imagen
def rectify(img, H):
    return cv.warpPerspective(img, H, (img.shape[1],img.shape[0]))


########################################################################
############  PROGRAMA PRINCIPAL  ######################################
########################################################################

print('\nstarting program\n')
for (key,frame) in autoStream():
    img = frame
    cv.imshow('original',img)

    # espera a que se marquen los puntos de referencia
    print('Click on the corners of the two squares')
    print('The corners must be marked in the same order in both squares\n')
    cv.setMouseCallback('original', mouse_callback)
    while numPtsReg < 2*numPts:
        cv.waitKey(1)
        # muestra la imagen
        cv.imshow('original',img)
        continue

    ptsC1 = np.array(ptsC1)
    ptsC2 = np.array(ptsC2)

    # máscara de los dos cuadrados marcados
    mC1 = cv.fillConvexPoly(np.zeros(img.shape[:2],np.uint8), ptsC1, 1)
    mC2 = cv.fillConvexPoly(np.zeros(img.shape[:2],np.uint8), ptsC2, 1)

    # calcula la matriz de transformación de C1 a C2
    H = calculateHomography(ptsC1, ptsC2)

    # y la inversa de C2 a C1
    IH = np.linalg.inv(H)

    # rectificamos el cuadrado C1 al plano de C2
    C1Rectified = rectify(img, H)
    # y nos quedamos solo con la máscara de C2
    C1Rectified = C1Rectified[mC2>0]
    # rectificamos el cuadrado C2 al plano de C1
    C2Rectified = rectify(img, IH)
    # y nos quedamos solo con la máscara de C1
    C2Rectified = C2Rectified[mC1>0]

    # muestra la imagen original sustituyendo los cuadrados por los rectificados
    imgSwapped = img.copy()
    imgSwapped[mC2>0] = C1Rectified
    imgSwapped[mC1>0] = C2Rectified
    
    # muestra la imagen original y la intercambiada
    cv.imshow('original',img)
    cv.imshow('swapped',imgSwapped)

    # espera a que se pulse una tecla
    cv.waitKey(0)
    cv.destroyAllWindows()








    


        
    
