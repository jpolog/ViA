#!/usr/bin/env python

import time
import numpy as np
import cv2 as cv

from umucv.util import ROI, putText,Video
from umucv.stream import autoStream

#Crea una ventana
cv.namedWindow("input")
#Ajusta la posición en la que se abrirá la ventana
cv.moveWindow('input', 200, 200)

#Crea un objeto roi de la pantalla, que detecta al ratón
region = ROI("input")

#Crea un substractor de fondos, al que si le pasamos un frame devolverá una máscara poniendo a 0 todo lo que considera fondo
#500 es el numero de fps que está teniendo en cuenta y 16 la diferencia permitida entre bytes antes de considerarlo un cambio
bgsub = cv.createBackgroundSubtractorMOG2(500, 16, False)

video = Video(fps=15, codec="MJPG",ext="avi")

recording = False

for key, frame in autoStream():
    #Si el ROI está definido
    if region.roi:
        #Esquinas de la región de interés
        [x1,y1,x2,y2] = region.roi
        #Crea una nueva pantalla que solo muestra la región de interés
        trozo = frame[y1:y2+1, x1:x2+1]
        cv.imshow("trozo", trozo)
        #Crea una máscara de movimineto (0 para los pixeles que no varian y 255 para los que si)
        fgmask = bgsub.apply(trozo)
        masked = trozo.copy()
        #Cada pixel del frame que la máscara detecta que no varia, lo convierte en negro (eliminamos el fondo)
        # fgmask==0 -> Pixel negro de fgmask
        # = 0 -> Lo convierte en un pixel negro 
        masked[fgmask==0] = 0
        #Si se ordena iniciar una grabación
        if (recording == False and key == ord('r')):
            t1 = time.time()
            #Comenzar a grabar
            video.write(masked,key,ord('r'))
            recording = True
        #Si está grabando
        if (recording == True):
            #Guardar el frame correspondiente
            video.write(masked)
            cv.circle(masked,(15,15),6,(0,0,255),-1)
            #Si llevamos más de 3 segundos grabando, detener la grabación y guardarla en nuestros archivos          
            if (time.time() - t1) > 3:
                video.write(masked,ord('a'),ord('a'))
                video.release()
                video = Video(fps=15, codec="MJPG",ext="avi")
                recording = False

        cv.imshow('object', masked)
        #Si pulsamos la x borramos el ROI
        if key == ord('x'):
            region.roi = []
            cv.destroyWindow("trozo")

        #Dibuja el ractángulo que conforma el ROI    
        cv.rectangle(frame, (x1,y1), (x2,y2), color=(0,255,255), thickness=2)
        putText(frame, f'{x2-x1+1}x{y2-y1+1}', orig=(x1,y1-8))

   
    #Devuelve la altura y anchura del frame
    h,w,_ = frame.shape
    putText(frame, f'{w}x{h}')
    cv.imshow('input',frame)
    
cv.destroyAllWindows()