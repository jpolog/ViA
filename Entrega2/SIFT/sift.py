#!/usr/bin/env python

# eliminamos muchas coincidencias erróneas mediante el "ratio test"

import cv2 as cv
import time

from umucv.stream import autoStream
from umucv.util import putText

sift = cv.SIFT_create(nfeatures=500)

matcher = cv.BFMatcher()

x0 = None

# alias para los elementos de los modelos
KEY = 0
DES = 1
IMG = 2

# umbral de coincidencias
PERC_THRESHOLD = 0.1

# lista de modelos
models = []

# función para leer los modelos
def readModels():
    import glob
    for fn in glob.glob('models/*.png'):
        print(fn)
        img = cv.imread(fn)
        k, d = sift.detectAndCompute(img, mask=None)
        models.append((k, d, img))

# enum para los estados
class State:
    IDLE, READ, MATCH = range(3)

state = State.IDLE

readModels()
    

for key, frame in autoStream():

    if key == ord('x'):
        x0 = None
        if state == State.MATCH:
            cv.destroyWindow('model')
            state = State.IDLE

    t0 = time.time()
    keypoints , descriptors = sift.detectAndCompute(frame, mask=None)
    t1 = time.time()
    putText(frame, f'{len(keypoints)} pts  {1000*(t1-t0):.0f} ms')

    if key == ord('c'):
        k0, d0, x0 = keypoints, descriptors, frame
        state = State.READ
        print('captured')

    if x0 is None:
        flag = cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        cv.imshow('SIFT', frame)
    else:
        t2 = time.time()
        if state == State.READ:
            bestPerc = 0
            bestModel = None
            bestMatches = []
            # solicitamos las dos mejores coincidencias de cada punto, no solo la mejor
            # seleccionamos el modelo que mejor coincida
            for i in range(len(models)):
                matches = matcher.knnMatch(models[i][DES], d0, k=2)
                
                good = []
                # ratio test
                # nos quedamos solo con las coincidencias que son mucho mejores que
                # que la "segunda opción". Es decir, si un punto se parece más o menos lo mismo
                # a dos puntos diferentes del modelo lo eliminamos.
                for m in matches:
                    if len(m) >= 2:
                        best,second = m
                        if best.distance < 0.75*second.distance:
                            good.append(best)

                if len(good)/len(matches) > bestPerc:
                    bestPerc = len(good)/len(matches)
                    bestModel = i
                    bestMatches = good
                
                print(f'{len(good)} matches with model {i}')


        t3 = time.time()

        cv.imshow("SIFT",frame)
        # si hay coincidencias suficientemente buenas
        if bestPerc > PERC_THRESHOLD and state == State.READ:
            state = State.MATCH
            # mostramos el modelo
            cv.imshow('model', models[bestModel][IMG])
            # dibujamos las coincidencias
            img3 = cv.drawMatches(models[bestModel][IMG], models[bestModel][KEY], x0, k0, bestMatches, None, flags=2)
            cv.imshow('matches', img3)
            print(f'model {bestModel} with {len(bestMatches)} matches')

        # no hay un modelo lo suficientemente bueno
        elif state == State.READ:
            state = State.IDLE

        
