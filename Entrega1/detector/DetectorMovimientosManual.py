#!/usr/bin/env python

import time
import numpy as np
import cv2 as cv

from umucv.util import ROI, putText, Video
from umucv.stream import autoStream

#Crea una ventana
cv.namedWindow("input")
#Ajusta la posición en la que se abrirá la ventana
cv.moveWindow('input', 200, 200)

#Crea un objeto roi de la pantalla (detecta al ratón al parecer)
region = ROI("input")
listFrames = []

video = Video(fps=15, codec="MJPG",ext="avi")

recording = False

for key, frame in autoStream():
    if (len(listFrames)==7):
        listFrames.pop(0)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    listFrames.append(gray)
    #Si el ROI está definido
    if region.roi:
        #Esquinas de la región de interés
        [x1,y1,x2,y2] = region.roi
        #Crea una nueva pantalla que solo muestra la región de interés
        trozo = frame[y1:y2+1, x1:x2+1]
        grayTrozo = cv.cvtColor(trozo, cv.COLOR_BGR2GRAY)
        oldTrozo = listFrames[0][y1:y2+1, x1:x2+1]
        dif = cv.absdiff(grayTrozo, oldTrozo)
        masked = trozo.copy()
        masked[dif<10] = 0
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

        cv.imshow("trozo", trozo)
        cv.imshow("oldTrozo",oldTrozo)
        cv.imshow("masked",masked)
        if key == ord('x'):
            region.roi = []
            cv.destroyWindow("oldTrozo")
            cv.destroyWindow("trozo")
            cv.destroyWindow("masked")

        #Dibuja el ractángulo que conforma el ROI 
        cv.rectangle(frame, (x1,y1), (x2,y2), color=(0,255,255), thickness=2)
        putText(frame, f'{x2-x1+1}x{y2-y1+1}', orig=(x1,y1-8))

   
    #DEvuelve la altura y anchura del frame
    h,w,_ = frame.shape
    putText(frame, f'{w}x{h}')
    cv.imshow('input',frame)
    
    
cv.destroyAllWindows()