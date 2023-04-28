#!/usr/bin/env python

import cv2 as cv
import numpy as np
from umucv.stream import autoStream
from umucv.util import putText
import time

tracks = []
track_len = 300
detect_interval = 10


corners_params = dict( maxCorners = 500,
                       qualityLevel= 0.1,
                       minDistance = 10,
                       blockSize = 7)

lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

first = True
sigma = 1 # gaussian blur

for n, (key, frame) in enumerate(autoStream()):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # gaussian blur
    gray = cv.GaussianBlur(gray, (0, 0), sigma)
    # mean filter
    gray = cv.blur(gray, (5, 5))
    cv.imshow('gaussian blur', gray)
    # threshold
    gray = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)

    # show the thresholded image
    cv.imshow('threshold', gray)
    
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
    if True:
        
        # Creamos una máscara para indicar al detector de puntos nuevos las zona
        # permitida, que es EL CUADRADO CENTRAL, quitando círculos alrededor de los puntos
        # existentes (los últimos de las trayectorias).
        # cuadrado central 50x50 en el centro de la imagen
        mask = np.zeros_like(gray)
        h,w = gray.shape
        # roi
        x1,x2,y1,y2 = h//2-25,h//2+25,w//2-25,w//2+25
        mask[x1:x2,y1:y2]= 255

        # print the roi
        cv.rectangle(frame, (y1,x1), (y2,x2), (0,255,0), 2)

        for x,y in [np.int32(t[-1]) for t in tracks]:
            # Estamos machacando con 0 en el radio de 5 alrededor de cada punto actual
            # para que no se repita ---> Buscar puntos en otra zona
            cv.circle(mask, (x,y), 5, 0, -1)
        cv.imshow("mask",mask)
        corners = cv.goodFeaturesToTrack(gray, mask=mask, **corners_params)
        if corners is not None:
            for [(x, y)] in np.float32(corners):
                tracks.append( [  [ x,y ]  ] )

        if len(tracks):
            # comprobamos si el punto está fuera del roi
            if tracks[0][-1][0] < y1 or tracks[0][-1][0] > y2 or tracks[0][-1][1] < x1 or tracks[0][-1][1] > x2:
                first = False

    putText(frame, f'{len(tracks)} corners, {(t1-t0)*1000:.0f}ms' )
    prevgray = gray

    cv.imshow('input', frame)

