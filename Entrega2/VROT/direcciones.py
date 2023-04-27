#!/usr/bin/env python

import cv2 as cv
import numpy as np
from umucv.stream import autoStream
from umucv.util import putText
from getangle import getAngle
import time

tracks = []
track_len = 20  
detect_interval = 5


corners_params = dict( maxCorners = 500,
                       qualityLevel= 0.1,
                       minDistance = 10,
                       blockSize = 7)

lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))


for n, (key, frame) in enumerate(autoStream()):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    t0 = time.time()
    if len(tracks):
        
        # el criterio para considerar bueno un punto siguiente es que si lo proyectamos
        # hacia el pasado, vuelva muy cerca del punto incial, es decir:
        # "back-tracking for match verification between frames"
        p0 = np.float32( [t[-1] for t in tracks] )
        p1,  _, _ =  cv.calcOpticalFlowPyrLK(prevgray, gray, p0, None, **lk_params)
        p0r, _, _ =  cv.calcOpticalFlowPyrLK(gray, prevgray, p1, None, **lk_params)
        d = abs(p0-p0r).reshape(-1,2).max(axis=1)
        good = d < 1
        
        new_tracks = []
        for t, (x,y), ok in zip(tracks, p1.reshape(-1,2), good):
            if not ok:
                continue
            t.append( [x,y] )
            if len(t) > track_len:
                del t[0]
            new_tracks.append(t)

        tracks = new_tracks


        # dibujamos las trayectorias
        cv.polylines(frame, [ np.int32(t) for t in tracks ], isClosed=False, color=(0,0,255))
        for t in tracks:
            x,y = np.int32(t[-1])
            cv.circle(frame, (x, y), 2, (0, 0, 255), -1)

    t1 = time.time()    

    
    # resetear el tracking
    if n % detect_interval == 0:
        
        # Creamos una máscara para indicar al detector de puntos nuevos las zona
        # permitida, que es toda la imagen, quitando círculos alrededor de los puntos
        # existentes (los últimos de las trayectorias).
        mask = np.zeros_like(gray)
        mask[:] = 255
        for x,y in [np.int32(t[-1]) for t in tracks]:
            # Estamos machacando con 0 en el radio de 5 alrededor de cada punto actual
            # para que no se repita ---> Buscar puntos en otra zona
            cv.circle(mask, (x,y), 5, 0, -1)
        #cv.imshow("mask",mask)
        corners = cv.goodFeaturesToTrack(gray, mask=mask, **corners_params)
        if corners is not None:
            for [(x, y)] in np.float32(corners):
                tracks.append( [  [ x,y ]  ] )

    putText(frame, f'{len(tracks)} corners, {(t1-t0)*1000:.0f}ms' )
    prevgray = gray

    movement = 'Sin movimiento' # por defecto
    if len(tracks) >= 2:
        # Extraemos las coordenadas de los puntos de las trayectorias
        corners_frames = [[pt for pt in track] for track in tracks]
        x_frames = [[pt[0] for pt in frame] for frame in corners_frames]
        y_frames = [[pt[1] for pt in frame] for frame in corners_frames]
        # Calculamos la media de los desplazamientos en cada frame (en cada eje)
        dx_frames = [np.mean(np.diff(frame)) for frame in x_frames]
        dy_frames = [np.mean(np.diff(frame)) for frame in y_frames]
        if dx_frames is None or dy_frames is None:
            continue
        # Estimamos la dirección del movimiento medio en cada eje
        dx_mean = np.mean(dx_frames)
        dy_mean = np.mean(dy_frames)
        if np.isnan(dx_mean) or np.isnan(dx_mean):
            movement = 'Sin movimiento'
            continue
        if abs(dx_mean) > abs(dy_mean): # Si el movimiento es más horizontal que vertical
            # Observación: la cámara se mueve en sentido contrario al movimiento del punto en la imagen
            if dx_mean > 0:
                movement = 'Izquierda'
            else:
                movement = 'Derecha'
        else: # Si el movimiento es más vertical que horizontal
            if dy_mean > 0:
                movement = 'Arriba'
            else:
                movement = 'Abajo'

        # dibujamos la dirección del movimiento como un vector desde el centro de la imagen
        h, w = frame.shape[:2]
        # se normaliza el vector de dirección y se escalan sus componentes para que se vea

        print(f'{movement} {dx_mean:.2f} {dy_mean:.2f}')

        # suponemos el vector medio de desplazamiento en el origen de coordenadas
        # (para hacer cálculos con el modelo pinhole de la cámara)
        start = [x_start, y_start]  = [w//2, h//2]
        end = [x_end, y_end] = [w//2 - int(dx_mean), h//2 - int(dy_mean)]

        # dibujamos el vector de dirección del movimiento
        # (para que se corresponda con el movimiento de la CÁMARA, no de la IMAGEN)
        cv.arrowedLine(frame, (start), (end), (0, 255, 0), 2)

        # Observación sobre los cálculos:
        # velocidad lineal
        #   linearVel = norm / track_len (px / frame)
        # grados por pixel 
        #   gpp = angle / norm (grados / px)
        #
        # --> Velocidad Angular = linearVel * gpp = angle / track_len (grados / frame)
        angle = getAngle(frame, start, end)

        # velocidad angular en grados / frame
        angVel = angle / track_len

        

        # con la información de la cámara, calculamos el ángulo de giro
        # usamos una de las trayectorias, tomando el primer y el último punto
        # (el primero es el más antiguo, el último el más reciente)
        # y calculamos el ángulo de giro entre ellos

        # dibujamos el ángulo de giro
        cv.putText(frame, f'{angle:.2f} deg.', (10, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)

        



        cv.imshow('input', frame)


    print('Camera movement:', movement)