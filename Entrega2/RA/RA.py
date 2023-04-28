#!/usr/bin/env python

# estimación de pose a partir del marcador images/ref.png
# En esta versión el objeto virtual se mueve.

# pruébalo con el vídeo de siempre

# ./pose2.py --dev=../../images/rot4.mjpg

# con la imagen de prueba

# ./pose2.py --dev=dir:../../images/marker.png

# o con la webcam poniéndolo en el teléfono o el monitor.

import cv2          as cv
import numpy        as np

from umucv.stream   import autoStream
from umucv.htrans   import htrans, Pose
from umucv.util     import cube, showAxes
from umucv.contours import extractContours, redu

# states of the cube
class State:
    STILL, MOVING = range(2)

# cube
SPEED = 0.5
class Cube:
    def __init__(self, size, pos, state, perspective):
        self.size = size
        self.pos = pos
        self.perspective = perspective
        self.state = state
        self.speed = SPEED
        self.path = [] # queue of points to follow

    def draw(self, frame):
        ### ARREGLAR
        cv.drawContours(frame, [cube(self.size, self.pos, self.perspective)], -1, (0,0,255), 2)

    def move(self):
        if self.state == State.MOVING:
            if len(self.path) > 0:
                self.pos = self.path.pop(0)
            else:
                self.state = State.STILL



def Kfov(sz,hfovd):
    hfov = np.radians(hfovd)
    f = 1/np.tan(hfov/2)
    # print(f)
    w,h = sz
    w2 = w / 2
    h2 = h / 2
    return np.array([[f*w2, 0,    w2],
                     [0,    f*w2, h2],
                     [0,    0,    1 ]])


stream = autoStream()

HEIGHT, WIDTH = next(stream)[1].shape[:2]
size = WIDTH,HEIGHT


K = Kfov( size, 60 )


marker = np.array(
       [[0,   0,   0],
        [0,   1,   0],
        [0.5, 1,   0],
        [0.5, 0.5, 0],
        [1,   0.5, 0],
        [1,   0,   0]])

square = np.array(
       [[0,   0,   0],
        [0,   1,   0],
        [1,   1,   0],
        [1,   0,   0]])



def polygons(cs,n,prec=2):
    rs = [ redu(c,prec) for c in cs ]
    return [ r for r in rs if len(r) == n ]

def rots(c):
    return [np.roll(c,k,0) for k in range(len(c))]

def bestPose(K,view,model):
    poses = [ Pose(K, v.astype(float), model) for v in rots(view) ]
    return sorted(poses,key=lambda p: p.rms)[0]



######################################
# Funciones para seguir la ruta ######
######################################

def binarize(gray):
    _, r = cv.threshold(gray, 128, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    return r


# Buffer de los últimos 30 frames tras binarize y threshold
skip = 3

# Detecta la ruta dibujada en la imagen
def detectPath(img, static):

    # binarizamos
    static = binarize(static)
    img = binarize(img)
    # diferenca entre la imagen actual y el modelo estatico
    diff = cv.absdiff(img, static)
    # threshold
    _, diff = cv.threshold(diff, 128, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    # extraemos los contornos, debe haber solo uno (el nuevo camino dibujado)
    cs = extractContours(diff, minarea=5, reduprec=2)
    good = redu(cs[0], 2)

    return good


def drawPath(img, path):
    cv.drawContours(img, [path], -1, (0,0,255), 3, cv.LINE_AA)
    return img

    



static = None

for n, (key,frame) in enumerate(stream):

    gray = binarize(frame)
    if static is None:
        static = gray
    cs = extractContours(gray, minarea=5, reduprec=2)

    good = polygons(cs,6,3)
    poses = []
    for g in good:
        p = bestPose(K,g,marker)
        if p.rms < 2:
            poses += [p.M]

    for M in poses:

        # capturamos el color de un punto cerca del marcador para borrarlo
        # dibujando un cuadrado encima
        x,y = htrans(M, (0.7,0.7,0) ).astype(int)
        b,g,r = frame[y,x].astype(int)
        cv.drawContours(frame,[htrans(M,square*1.1+(-0.05,-0.05,0)).astype(int)], -1, (int(b),int(g),int(r)) , -1, cv.LINE_AA)
        # cv.drawContours(frame,[htrans(M,marker).astype(int)], -1, (0,0,0) , 3, cv.LINE_AA)

        # creamos el cubo en la posición del marcador
        cube = Cube(0.2, htrans(M, (0.5,0.5,0)), State.MOVING)

        # Mostramos el sistema de referencia inducido por el marcador (es una utilidad de umucv)
        showAxes(frame, M, scale=0.5)

        # hacemos que se mueva el cubo
        cosa = cube * (0.5,0.5,0.75 + 0.5*np.sin(n/100)) + (0.25,0.25,0)
        cv.drawContours(frame, [ htrans(M, cosa).astype(int) ], -1, (0,128,0), 3, cv.LINE_AA)


        detectPath(frame,gray)

        

    cv.imshow('source',frame)
    