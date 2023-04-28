import cv2 as cv
import numpy as np
from umucv.stream import autoStream
from umucv.util import putText
from scipy.ndimage import minimum_filter, maximum_filter

#Se crea la ventana principal y la ventana de ayuda
cv.namedWindow("help")
cv.namedWindow("input")

#Añadimos trackbars para controlarcada uno de los filtros a nuestro antojo
BOX = [40]
cv.createTrackbar('boxVariable', 'input', BOX[0], 100, lambda v: BOX.insert(0,v))

GAU = [3]
cv.createTrackbar('gaussianVariable', 'input', GAU[0]*10, 100, lambda v: GAU.insert(0,v/10))

MED = [3]
cv.createTrackbar('medianVariable', 'input', int(MED[0]/2), 20, lambda v: MED.insert(0,v*2+1))

BIL = [10]
cv.createTrackbar('bilateralVariable', 'input', BIL[0]*5, 100, lambda v: BIL.insert(0,v/5))

MIN = [17]
cv.createTrackbar('minimumVariable', 'input', MIN[0], 100, lambda v: MIN.insert(0,v))

MAX = [17]
cv.createTrackbar('maximumVariable', 'input', MAX[0], 100, lambda v: MAX.insert(0,v))

#Booleano que controla la ventana HELP
help = True

#Booleano que controla si los filtros se aplican a todo el frame o solo una porción
applyToPortion = True

#Todas las teclas que producen un cambio de filtro
importantKeys = [ord('0'),ord('1'),ord('2'),ord('3'),ord('4'),ord('5'),ord('6')]

#Lista que indica para cada filtro posible, si ha sido aplicado (apartado opcional e) )
#El primer elemento estará a True cuando no haya ningun filtro seleccionado (0: do nothing)
listApplied = [True,False,False,False,False,False,False]

#Gestiona la lista de filtros seleccionados
def filterSelection(key):
    #Si se ha pulsado el 0 queremos que no se aplique ningún filtro
    if key==(ord('0')):
        for item in listApplied:
            item = False
        listApplied[0] = True
    #Si se ha pulsado la tecla para cualquier filtro, guardar el cambio en la lista
    else:
        listApplied[0] = False
        if key ==(ord('1')):
            listApplied[1] = not listApplied[1]
        elif key ==(ord('2')):
            listApplied[2] = not listApplied[2]
        elif key ==(ord('3')):
            listApplied[3] = not listApplied[3]
        elif key ==(ord('4')):
            listApplied[4] = not listApplied[4]
        elif key ==(ord('5')):
            listApplied[5] = not listApplied[5]
        else:
            listApplied[6] = not listApplied[6]


for key,frame in autoStream():
    #Se crea un fondo negro y sobre el se escribe toda la información de la ventana HELP
    fondoNegro = np.zeros((320, 320, 3), np.uint8)
    putText(fondoNegro,"BLUR FILTERS", orig=(5,35), scale = 2)
    putText(fondoNegro,"0: do nothing", orig=(5,70))
    putText(fondoNegro,"1: box", orig=(5,90))
    putText(fondoNegro,"2: Gaussian", orig=(5,110))
    putText(fondoNegro,"3: median", orig=(5,130))
    putText(fondoNegro,"4: bilateral", orig=(5,150))
    putText(fondoNegro,"5: min", orig=(5,170))
    putText(fondoNegro,"6: max", orig=(5,190))
    putText(fondoNegro,"r: only roi", orig=(5,240))
    putText(fondoNegro,"h: show/hide help", orig=(5,290))

    #Si la tecla pulsada es la h, abrir o cerrar la ventana HELP
    if key == (ord('h')):
        help = not help
        if not help: cv.destroyWindow('help')

    #Si la tecla pulsada es la r, aplicar los filtros a todo el frame o a una sola porción
    elif key == (ord('r')):
        applyToPortion = not applyToPortion
    
    #Se aplica el filtro a una porcion del frame
    if applyToPortion: portion = frame[200:400, 300:500]
    #Se aplica el filtro a todo el frame
    else: portion = frame
    
    #Si se ha pulsado una tecla de filtro, gestionar la lista de filtros aplicados
    if (key in importantKeys):
        filterSelection(key)
    

    #Si se debe aplicar al menos un filtro
    if (not listApplied[0]): 
        #Transformamos la porcion en la que queremos aplicar el filtro a escala de grises
        portion = cv.cvtColor(portion,cv.COLOR_BGR2GRAY)
        #BoxFilter
        if listApplied[1]:
            portion = cv.boxFilter(portion, -1, (BOX[0], BOX[0])) if BOX[0] > 0 else portion.copy()
            putText(frame,f"1. boxVariable = {BOX[0]}", orig = (5,35))
        #GaussianBlur
        if listApplied[2]:
            portion = cv.GaussianBlur(portion, (0,0), GAU[0]) if GAU[0] > 0 else portion.copy()
            putText(frame,f"2. gaussianVariable = {GAU[0]}", orig = (5,55))
        #Median
        if listApplied[3]:
            portion =cv.medianBlur(portion,MED[0])
            putText(frame,f"3. medianVariable = {MED[0]}", orig = (5,75))

        #Bilateral
        if listApplied[4]:
            portion =cv.bilateralFilter(portion,0,BIL[0],BIL[0])    
            putText(frame,f"4. bilateralVariable = {BIL[0]}", orig = (5,95))

        #Minimum
        if listApplied[5]:
            portion =minimum_filter(portion, size=MIN[0])
            putText(frame,f"5. minimumVariable = {MIN[0]}", orig = (5,115))

        #Maximum
        if listApplied[6]:
            portion =maximum_filter(portion,size=MAX[0])
            putText(frame,f"6. maximumVariable = {MAX[0]}", orig = (5,135))

        #Reemplazar los bytes del frame correspondiente a la porción, 
        #por esta porción con el flitro correspondiente aplicado
        if applyToPortion: frame[200:400, 300:500] = cv.merge((portion,portion,portion))
        else: frame = portion

    #Mostrar el frame por pantalla
    cv.imshow('input',frame)
    #Mostrar la ventana HELP si está activada
    if help:
        cv.imshow('help',fondoNegro)


