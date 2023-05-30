#!/usr/bin/env python


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
detect_interval = 20
corners_params = dict( maxCorners = 500,
                    qualityLevel= 0.1,
                    minDistance = 10,
                    blockSize = 7)
lk_params = dict( winSize  = (15, 15),
                maxLevel = 2,
                criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
first = True



MOV_DELAY = 10  # se mueve 1 de cada 10 frames

# states of the cube
class State:
    STILL, FINDING, FOUND, MOVING = range(4)

# cube
class Cube:
    def __init__(self, size, pos, state, pose):
        self.size = size
        # homogeneous coordinates
        self.position = np.array([pos[0], pos[1]])   # coordenadas real del marcador
        self.pose = pose # matrix to transform from 3D to 2D
        self.state = state
        self.motion = 0
        # empty array of points
        self.path = np.empty((0,2), dtype=int)
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
                hom = np.delete(self.pose, 2, 1)
                I = np.linalg.inv(hom)
                track = tracks[0][len(self.path):]
                if len(self.path) == 0:
                    self.path = htrans(I,[p for p in track])
                    print("firstPath",self.path)
                elif len(track) > 20:
                    test = [p for p in track]
                    print("test",test)
                    followingPath = htrans(I,test)
                    print("followingPath",followingPath)
                    self.path = np.concatenate((self.path,followingPath))
                
            # dibujamos las trayectorias
            cv.polylines(frame, [ np.int32(t) for t in tracks ], isClosed=False, color=(0,0,255))
            for t in tracks:
                x,y = np.int32(t[-1])
                cv.circle(frame, (x, y), 2, (0, 0, 255), -1)
   

        
        # resetear el tracking
        if self.state == State.FINDING:
            
            # Creamos una máscara para indicar al detector de puntos nuevos las zona
            # permitida, que es EL CUADRADO abajo a la izquierda, quitando círculos alrededor de los puntos
            # existentes (los últimos de las trayectorias).
            # cuadrado central 50x50 en el centro de la imagen
            mask = np.zeros_like(gray)
            h,w = gray.shape
            # roi
            x1,x2,y1,y2 = h//3-25,h//3+25,2*w//3-25,2*w//3+25

            # aplicamos transformación de perspectiva para que el cuadrado esté en el plano del marcador
            H = np.delete(self.pose, 2, 1)
            [x1,y1] = htrans(H,[x1,y1])
            [x2,y2] = htrans(H,[x2,y2])   

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
        H = np.delete(self.pose, 2, 1)
        imgPath = np.int32(htrans(H, self.path))
        cv.polylines(img, [imgPath], isClosed=False, color=(0,255,0), thickness=3)
            

    def move(self):
        if self.state == State.MOVING:
            if self.motion == MOV_DELAY:
                if len(self.path) > skip:
                    self.position = [self.path[0][0], self.path[0][1]]
                    self.path = self.path[skip:]
                elif len(self.path) > 0:
                    self.position = [self.path[-1][0], self.path[-1][1]]
                    self.path = []
                else:
                    self.state = State.STILL
            else:
                self.motion += 1

        print(self.position)


    def draw(self, frame):
        cornersToPrint = np.array([c+[self.position[0], self.position[1],0] for c in self.corners])
        # transform
        pts = htrans(M, cornersToPrint).astype(int)
        #print(pts)
        pts = np.array(pts).astype(int)
        #print(pts)
        cv.drawContours(frame, [pts], -1, (0,128,0), 3, cv.LINE_AA)
        




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

    if len(poses):
        M = poses[0] # la mejor pose
    else:
        cv.imshow('source',frame)
        continue

    # capturamos el color de un punto cerca del marcador para borrarlo
    # dibujando un cuadrado encima
    x,y = htrans(M, (0.7,0.7,0) ).astype(int)
    b,g,r = frame[y,x].astype(int)
    cv.drawContours(frame,[htrans(M,square*1.1+(-0.05,-0.05,0)).astype(int)], -1, (int(b),int(g),int(r)) , -1, cv.LINE_AA)
    # cv.drawContours(frame,[htrans(M,marker).astype(int)], -1, (0,0,0) , 3, cv.LINE_AA)
    if cube is None:
        # creamos el cubo en la posición del marcador
        cube = Cube(0.2, [0.5,0.5,0], State.STILL, M)
    else:
        cube.pose = M
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
    
    #controles manuales
    if key == ord('c'):
        cube.state = State.FINDING
    if key == ord('v'):
        cube.state = State.FOUND
    if key == ord('m'):
        cube.state = State.MOVING
    
    # Acciones según el estado del cubo
    if cube.state == State.STILL:
        
        print("still")
    elif cube.state == State.FINDING:
        #print("detecting path")
        cube.detectPath()

    elif cube.state == State.FOUND:
        cube.drawPath(frame)
        #cube.state = State.MOVING
        #cube.drawPath(frame, good)
        if cube.motion == 2*MOV_DELAY:
            cube.state = State.MOVING
            cube.motion = 0

    elif cube.state == State.MOVING:
        cube.move()
    
        print("moving")


    prevgray = gray

    cube.draw(frame)
    cv.imshow('source',frame)

    