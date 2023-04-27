#!/usr/bin/env python

import cv2          as cv
from umucv.stream import autoStream
import numpy as np
# la fft de numpy
from numpy.fft import fft

def binarize(gray):
    _, r = cv.threshold(gray, 128, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    return r
    
def extractContours(image):
    g = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    if black:
        g = 255-g
    b = binarize(g)  
    contours, _ = cv.findContours(b.copy(), cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)
    contours = [ c.reshape(-1,2) for c in contours ]
    contours = sorted(contours, key=cv.contourArea, reverse=True)
    return contours


###> AJUSTAR ESTOS PARÁMETROS PARA QUE FUNCIONE BIEN
###> en función de la resolución de la imagen  
def razonable(c):
    return 10000**2 >= cv.contourArea(c) >= 10**2


def orientation(x):
    return cv.contourArea(x.astype(np.float32),oriented=True) >= 0

black = True

# para elegir si mostramos la imagen original o todos los contornos
shcont = True

#######################################################################
def redu(c, eps=0.5):
    red = cv.approxPolyDP(c,eps,True)
    return red.reshape(-1,2)

def polygons(cs,n,prec=2):
    rs = [ redu(c,prec) for c in cs ]
    quad = []
    other = []
    for i in range(len(rs)):
        if len(rs[i]) == n:
            quad.append(rs[i])
        else:
            other.append(cs[i])
    return quad, other

def findSudokuContours(quad_borders, quad_interior, contour_numbers, mode):
    
    # Recorremos los contornos de borde, y nos quedamos 
    # con el contorno de mayor área
    maxArea = 0
    maxC = None
    currentC = -1   #count the number of contours
    for c in quad_borders:
        currentC += 1
        area = cv.contourArea(c)
        if area > maxArea:
            maxArea = area
            maxC = currentC
    
    
    if mode == 0:   # solo detectamos el cuadrado exterior de mayor área (borde del sudoku)
        return maxC, None, None

        
    elif mode == 1: # detectamos todos los contornos dentro del sudoku

        # Buscamos los contornos interiores dentro del cuadrado de mayor área. Debe haber 81
        # numpy arrays de contornos interiores y números dentro de los contornos interiores
        inside = np.empty(81, dtype=list)
        numbers_inside = np.full(81,None, dtype=object)


        j = 0   # index of inside matrix
        for c in range(len(quad_interior)):
            in_points = 0
            ### Cambiar esto por un for-else
            for i in range(len(quad_interior[c])):
                # comprueba si el punto está completamente dentro del cuadrado de mayor área
                if cv.pointPolygonTest(quad_borders[maxC], quad_interior[c][i].tolist(), False) == 1:
                    in_points += 1
            # si todos los puntos están dentro del cuadrado de mayor área
            # se añade a la lista de contornos interiores
            if in_points == len(quad_interior[c]):
                inside[j] = quad_interior[c]            
                # para cada contorno de numeros, 
                # se comprueba si está dentro del contorno
                for n in contour_numbers:
                    if cv.pointPolygonTest(quad_interior[c], n[0].tolist(), False) == 1:
                        # si está dentro, se añade a la lista de números de dentro del cuadrado
                        numbers_inside[j]= n
                        break
                # if not, a None is added
                else:
                    pass
                
                j = j+1
                if j==81:    # máximo de contornos interiores del sudoku
                    break

        # inside tiene que tener 81 contornos
        if len(inside) != 81:
            return None, None, None
        

        # reordena inside por la posición de sus esquinas en el eje y
        # reordena numbers_inside por el mismo orden que inside
        # Ambas matrices tiene las mismas dimensiones
        idx = np.argsort([polygon[0][1] for polygon in inside])
        inside = inside[idx]
        numbers_inside = numbers_inside[idx]
        # y luego de 9 en 9 por la posición de sus esquinas en el eje x
        for i in range(0,len(inside),9):
            idx = np.argsort([polygon[0][0] for polygon in inside[i:i+9]])
            inside[i:i+9] = inside[i:i+9][idx]
            numbers_inside[i:i+9] = numbers_inside[i:i+9][idx]



        
        # retornamos el índice del contorno de borde más grande
        #  y los contornos interiores
    
    
    return maxC, inside, numbers_inside
        
        
#######################################################################
############  MODELOS DE RECONOCIMIENTO DE NÚMEROS  ###################
#######################################################################

# basándonos en imágenes de los números de 0 a 9
# creamos un modelo de reconocimiento de números
# a partir de los contornos de los números
def createModels():
    # creamos un diccionario con los contornos de los números
    models = []
    for i in range(9):
        img = cv.imread('numbers/{}.png'.format(i+1))
        contours = extractContours(img)
        models.append(contours[0])
    return models

# creamos los modelos de los números
numberModels = createModels()

# Definimos la función que calcula el invariante
# de forma basado en las frecuencias dominantes
# esta función recibe un contorno y produce descriptor de tamaño wmax*2+1
def invar(c, wmax=10):
    x,y = c.T
    z = x+y*1j
    f  = fft(z)
    fa = abs(f)

    s = fa[1] + fa[-1]

    v = np.zeros(2*wmax+1)
    v[:wmax] = fa[2:wmax+2]
    v[wmax:] = fa[-wmax-1:]

   
    if fa[-1] > fa[1]:
        v[:-1] = v[-2::-1]
        v[-1] = fa[1]

    return v / s

# comparamos los invariantes de los contornos 
# encontrados en la imagen con el modelo y señalamos
# los que son muy parecidos
MAXDIST = 0.08

# como el invariante del 6 y del 9 son muy parecidos
# (asumimimos que el 6 es un 9 rotado)
# definimos una función que comprueba si el contorno
# es un 6 o un 9, calculando la distribución de píxeles
# en el contorno
def is_6_or_9(contour):
    # centroide del contorno
    M = cv.moments(contour)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    

    # si es un 6, el centroide estará en la parte superior
    # si es un 9, el centroide estará en la parte inferior
    # calculamos la distribución de píxeles en el contorno
    x, y, w, h = cv.boundingRect(contour)
    center_y = y + h / 2
    if cy > center_y:
        return 6
    else:
        return 9


def identifyNumbers(contours, models):
    #almacenamos los números encontrados en su posición del sudoku
    # array de 81 posiciones a 0
    numbers = [0]*81
    gaps = [n for n in range(81)]    # los huecos que hay que rellenar
    lenNumbers = 0
    j = 0

    for m in range(len(models)):
        j+=1
        
        invmodel = invar(models[m])
        for i in range(len(contours)):
            c = contours[i]
            if c is None:
                continue
            if c is not None:
                if np.linalg.norm(invar(c)-invmodel) < MAXDIST:
                    if m == 5 or m == 8:
                        number = is_6_or_9(contours[i]) -1
                        if m != number: # ha detectado un 6 como 9 o viceversa
                            continue

                    # se sustituye el contorno por el número
                    numbers[i] = m+1
                    lenNumbers += 1
                    gaps.remove(i)    # se elimina el hueco
        
            

    # si todos los números han sido identificados
    # se devuelve la lista de números
    lenContours = sum(1 for i in contours if i is not None)
    if lenNumbers == lenContours:
        # se devuelve la lista de números
        # en un array de 9x9
        sudoku = np.array(numbers).reshape(9,9)
        return sudoku, gaps
    else:
        return None, None


########################################################################
############  RESOLUCIÓN DEL SUDOKU  ###################################
########################################################################
#from sudoku_solver import solve_sudoku

def solve_sudoku(puzzle):
    """
    Solve a Sudoku puzzle given in a 9x9 NumPy array.
    Returns the solved puzzle as a 9x9 NumPy array.
    """
    # Helper function to find empty cells in the puzzle
    def find_empty_cell(puzzle):
        for row in range(9):
            for col in range(9):
                if puzzle[row][col] == 0:
                    return (row, col)
        return None
    
    # Helper function to check if a value can be placed in a cell
    def is_valid(puzzle, row, col, value):
        # Check row
        if value in puzzle[row]:
            return False
        
        # Check column
        if value in puzzle[:,col]:
            return False
        
        # Check box
        box_row = (row // 3) * 3
        box_col = (col // 3) * 3
        box = puzzle[box_row:box_row+3, box_col:box_col+3]
        if value in box:
            return False
        
        return True
    
    # Helper function to solve the puzzle using backtracking
    def solve_backtrack(puzzle):
        # Find the next empty cell
        empty_cell = find_empty_cell(puzzle)
        
        # If there are no empty cells, the puzzle is solved
        if not empty_cell:
            return True
        
        # Try values from 1 to 9 in the empty cell
        row, col = empty_cell
        for value in range(1, 10):
            # Check if the value is valid in the cell
            if is_valid(puzzle, row, col, value):
                # Place the value in the cell
                puzzle[row][col] = value
                
                # Recursively solve the rest of the puzzle
                if solve_backtrack(puzzle):
                    return True
                
                # If the puzzle cannot be solved with the current value, backtrack
                puzzle[row][col] = 0
        
        # If no value can be placed in the cell, the puzzle is unsolvable
        return False
    
    # Make a copy of the puzzle so the original is not modified
    puzzle_copy = puzzle.copy()
    
    # Solve the puzzle using backtracking
    solve_backtrack(puzzle_copy)
    
    return puzzle_copy


def display_sudoku(sudoku):
    # Check if input is a valid Sudoku array
    if sudoku.shape != (9, 9):
        print("Invalid Sudoku array.")
        return

    # Loop through each row and column to print the Sudoku grid
    for row in range(9):
        if row % 3 == 0 and row != 0:
            print("------+-------+------")
        for col in range(9):
            if col % 3 == 0 and col != 0:
                print("|", end=" ")
            print(sudoku[row][col], end=" ")
        print()


########################################################################
############  HOMOGRAFÍA  ##############################################
########################################################################
from umucv.htrans import htrans, desp, scale

marker = np.array(
       [[1,0],
        [0,0],
        [0, 1],
        [1,1]])

rectified = False

########################################################################
############  PROGRAMA PRINCIPAL  ######################################
########################################################################


found = False 
print('\nstarting program\n')
for (key,frame) in autoStream():
    result = np.zeros_like(frame)
    
    if key == ord('c'):
        shcont = not shcont

    contours = extractContours(frame)
    contours_borders = [c for c in contours if razonable(c) and not orientation(c)]

    quad_borders_orig, contour_numbers = polygons(contours_borders,n=4,prec=3)

    if not found:
        maxC, _, _ = findSudokuContours(quad_borders_orig, None, None, mode=0)
        if maxC is not None:
            found = True
            print('Sudoku border found')

    if found:
        if not rectified:
            corners = quad_borders_orig[maxC]
            for i in range(len(corners)):
                cv.circle(result, tuple(corners[i]), 5, (255,255,255), -1)

            H,_ = cv.findHomography(corners, marker)

            T = desp([100,100]) @ scale([500,500]) @ H
            rectif = cv.warpPerspective(frame, T, (800,800))
            cv.imshow('rectif', rectif)

            IH = np.linalg.inv(T)
            horiz = np.array( [ [[x,100],[x,600]] for x in np.arange(100,600,500*(1/9))] )
            vert  = np.array( [ [[100,y],[600,y]] for y in np.arange(100,600,500*(1/9))] )

            thoriz = htrans(IH, horiz.reshape(-1,2)).reshape(-1,2,2)
            tvert  = htrans(IH,  vert.reshape(-1,2)).reshape(-1,2,2)

            cv.polylines(frame, thoriz.astype(int), False, (255,0,255), 2, cv.LINE_AA) 
            cv.polylines(frame,  tvert.astype(int), False, (255,0,255), 2, cv.LINE_AA) 

            rectified = True
            found = False

            corners = np.round(htrans(T, corners)).astype(int)
            rectif_crop = rectif[corners[1][0]:corners[0][0], corners[0][1]:corners[3][1]]
        
        contours_rect = extractContours(rectif_crop)

        contours_borders = [c for c in contours_rect if razonable(c) and not orientation(c)]
        contours_interior = [c for c in contours_rect if razonable(c) and orientation(c)]

