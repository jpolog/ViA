#!/usr/bin/env python

# estimación de pose a partir del marcador images/ref.png
# En esta versión el objeto virtual se mueve.

# pruébalo con el vídeo de siempre

# ./pose2.py --dev=../../images/rot4.mjpg

# con la imagen de prueba

# ./pose2.py --dev=dir:../../images/marker.png

# o con la webcam poniéndolo en el teléfono o el monitor.

from collections import deque
import cv2          as cv
import numpy        as np

from umucv.stream   import autoStream
from umucv.htrans   import htrans, Pose
from umucv.util     import showAxes
from umucv.contours import extractContours, redu


test = True



# Detecta la ruta dibujada en la imagen
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





# states of the cube
class State:
    STILL, FINDING, FOUND, MOVING = range(4)

# cube
SPEED = 0.5
class Cube:
    def __init__(self, size, pos, state, pose):
        self.size = size
        # homogeneous coordinates
        self.position = np.array([pos[0], pos[1], 1])   # coordenadas homogeneas en el plano real para realizar los calculos
        self.pose = pose # matrix to transform from 3D to 2D
        self.state = state
        self.speed = SPEED
        self.path = [] # queue of points to follow
        self.corners = np.array([[0,0,0],
                                 [1,0,0],
                                 [1,1,0],
                                 [0,1,0],
                                 [0,0,0],
                                 
                                 [0,0,1],
                                 [1,0,1],
                                 [1,1,1],
                                 [0,1,1],
                                 [0,0,1],
                                     
                                 [1,0,1],
                                 [1,0,0],
                                 [1,1,0],
                                 [1,1,1],
                                 [0,1,1],
                                 [0,1,0]
                                 ])




    def detectPath(self):
        global tracks, track_len, detect_interval, corners_params, lk_params, first
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
                if np.linalg.norm(np.array([x,y])-t[-1]) < 1:
                    continue
                t.append( [x,y] )
                if len(t) > track_len:
                    del t[0]
                new_tracks.append(t)
                
            if len(new_tracks) > 0:
                tracks = new_tracks
            else: # se ha parado de dibujar el camino ---> Camino fijo
                self.state = State.FOUND
                # se almacena el track más largo como la trayectoria transformado a coordenadas reales
                self.path = htrans(self.pose,[[p[0],p[1],0] for p in tracks[0]]).astype(int)
                
            # dibujamos las trayectorias
            cv.polylines(frame, [ np.int32(t) for t in tracks ], isClosed=False, color=(0,0,255))
            for t in tracks:
                x,y = np.int32(t[-1])
                cv.circle(frame, (x, y), 2, (0, 0, 255), -1)
   

        
        # resetear el tracking
        if self.state == State.FINDING:
            
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
            corners = cv.goodFeaturesToTrack(gray, mask=mask, **corners_params)
            if corners is not None:
                for [(x, y)] in np.float32(corners):
                    tracks.append( [  [ x,y ]  ] )

            if len(tracks):
                # comprobamos si el punto está fuera del roi
                if tracks[0][-1][0] < y1 or tracks[0][-1][0] > y2 or tracks[0][-1][1] < x1 or tracks[0][-1][1] > x2:
                    first = False
        
            
            return tracks

    

    def drawPath(self,img):
        # delete the z coordinate of the pose matrix
        hom = np.delete(self.pose, 2, 1)
        I = np.linalg.inv(hom)
        imgPath = np.int32(htrans(I, self.path))
        cv.polylines(img, [imgPath], isClosed=False, color=(0,255,0), thickness=3)
            

    def move(self):
        if self.state == State.MOVING:
            if len(self.path) > skip:
                self.position = np.vstack([self.path[0][0], self.path[0][1], 1])
                self.path = self.path[skip:]
            elif len(self.path) > 0:
                self.position = np.vstack([self.path[-1][0], self.path[-1][1], 1])
                self.path = []
            else:
                self.state = State.STILL

        print(self.position)


    def draw(self, frame):
        # move corners depending on cube position using homogeneous coordinates
        T = np.eye(4)
        T[:3,3:4] = np.vstack([self.position[0], self.position[1], 0]) # we move 0 in z axis
        # corners to homogeneous coordinates
        corners = np.vstack([self.corners, [1,1,1]])
        print(corners)
        # transform
        corners = np.array(np.dot(T, corners)[:3]).astype(np.int32)
        print(corners)
        cv.drawContours(frame, [ htrans(self.pose, corners).astype(int) ], -1, (0,128,0), 3, cv.LINE_AA)




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
sigma = 1 # gaussian blur
def binarize(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, gray = cv.threshold(gray, 128, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    return gray


    
# Buffer de los últimos 30 frames tras binarize y threshold
skip = 3
buf = deque(maxlen=10)



cube = None
for n, (key,frame) in enumerate(stream):

    gray = binarize(frame)
    buf.append(gray)


    cs = extractContours(gray, minarea=5, reduprec=2)

    good = polygons(cs,6,3)
    poses = []
    for g in good:
        p = bestPose(K,g,marker)
        if p.rms < 2:
            poses += [p.M]

    # dejar solamente la mejor pose??
    for M in poses:

        # capturamos el color de un punto cerca del marcador para borrarlo
        # dibujando un cuadrado encima
        x,y = htrans(M, (0.7,0.7,0) ).astype(int)
        b,g,r = frame[y,x].astype(int)
        cv.drawContours(frame,[htrans(M,square*1.1+(-0.05,-0.05,0)).astype(int)], -1, (int(b),int(g),int(r)) , -1, cv.LINE_AA)
        # cv.drawContours(frame,[htrans(M,marker).astype(int)], -1, (0,0,0) , 3, cv.LINE_AA)

        if cube is None:
            # creamos el cubo en la posición del marcador
            cube = Cube(0.2, [0.5,0.5,0], State.STILL, M)

        # Mostramos el sistema de referencia inducido por el marcador (es una utilidad de umucv)
        showAxes(frame, M, scale=0.5)

        # hacemos que se mueva el cubo
        #cosa = cube * (0.5,0.5,0.75 + 0.5*np.sin(n/100)) + (0.25,0.25,0)
        #cv.drawContours(frame, [ htrans(M, cosa).astype(int) ], -1, (0,128,0), 3, cv.LINE_AA)

        # comparamos el frame actual con el más antiguo del buffer
        # para saber si se está dibujando un nuevo camino
        diff1 = cv.absdiff(gray, buf[0]).mean()
        diff2 = cv.absdiff(gray, buf[len(buf)//2]).mean()
        diff = (diff1 + diff2) / 2

        if key == ord('c'):
            cube.state = State.FINDING

        if key == ord('v'):
            cube.state = State.FOUND
        

        if cube.state == State.FINDING or cube.state == State.FOUND:
            #print("detecting path")
            cube.detectPath()
            if cube.state == State.FOUND:
                cube.drawPath(frame)
                #cube.state = State.MOVING
                #cube.drawPath(frame, good)

        if key == ord('m'):
            cube.state = State.MOVING

        if cube.state == State.MOVING:
            cube.move()
            cube.draw(frame)
            cv.polylines(frame, [cube.path], False, (0,255,0), 3, cv.LINE_AA)



    prevgray = gray

    cv.imshow('source',frame)

    