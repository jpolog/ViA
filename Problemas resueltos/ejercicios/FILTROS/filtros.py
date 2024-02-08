import cv2 as cv
import numpy as np
from umucv.stream import autoStream
from umucv.util import putText
from scipy.ndimage import minimum_filter, maximum_filter
from umucv.util   import ROI

# Create two windows: 'help' and 'input'
cv.namedWindow("help")
cv.namedWindow("input")
region = ROI('input')

# Create trackbars to adjust the parameters of each filter
box_value = [40]
cv.createTrackbar('boxVariable', 'input', box_value[0], 100, lambda v: box_value.insert(0,v))

gauss_value = [3]
cv.createTrackbar('gauss_valuessianVariable', 'input', gauss_value[0], 50, lambda v: gauss_value.insert(0,v))

median_value = [3]
cv.createTrackbar('medianVariable', 'input', int(median_value[0]/2), 25, lambda v: median_value.insert(0,v*2+1))

bilateral_value = [10]
cv.createTrackbar('bilateralVariable', 'input', bilateral_value[0], 20, lambda v: bilateral_value.insert(0,v))

minimum_value = [17]
cv.createTrackbar('minimumVariable', 'input', minimum_value[0], 100, lambda v: minimum_value.insert(0,v))

maximum_value = [17]
cv.createTrackbar('maximumVariable', 'input', maximum_value[0], 100, lambda v: maximum_value.insert(0,v))

# Boolean variables to control the windows and filters
show_help = True
apply_to_portion = True
# whether each filter is applied or not
list_applied = [True,False,False,False,False,False,False]

change_keys = [ord('0'),ord('1'),ord('2'),ord('3'),ord('4'),ord('5'),ord('6')]
# Function to manage the applied filters list
def filter_selection(key):
    # If the key is 0, do nothing
    if key == ord('0'):
        for i in range(len(list_applied)):
            list_applied[i] = False
        list_applied[0] = True
    else:
        # If the key is not 0, apply the selected filter
        list_applied[0] = False
        index = key - ord('1') + 1
        if 0 < index < len(list_applied):
            list_applied[index] = not list_applied[index]

for key, frame in autoStream():

    # Create a black background and write the information about the available filters
    help_window = np.zeros((360, 320, 3), np.uint8)
    putText(help_window,"BLUR FILTERS", orig=(5,35), scale = 2)
    putText(help_window,"KEYS:", orig=(5,55))
    putText(help_window,"0: do nothing", orig=(5,70))
    putText(help_window,"1: box", orig=(5,90))
    putText(help_window,"2: Gaussian", orig=(5,110))
    putText(help_window,"3: median", orig=(5,130))
    putText(help_window,"4: bilateral", orig=(5,150))
    putText(help_window,"5: min", orig=(5,170))
    putText(help_window,"6: max", orig=(5,190))
    putText(help_window,"r: only roi", orig=(5,240))
    putText(help_window,"h: show/hide help", orig=(5,290))

    # Toggle the help window when the 'h' key is pressed
    if key == ord('h'):
        show_help = not show_help
        if not show_help: cv.destroyWindow('help')

    # Toggle whether to apply the filters to the whole frame or just a portion
    elif key == ord('r'):
        apply_to_portion = not apply_to_portion
    

    # Apply filters to the selected portion of the frame
    if apply_to_portion:
        if region.roi:
            x1,y1,x2,y2 = region.roi
            # If the region is not valid, show the help window and the input frame
            # without applying any filter
            if (x2-x1)*(y2-y1) == 0 or (x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]): 
                cv.imshow('help',help_window)
                cv.imshow('input',frame)
                continue
        else:
            # If no region is selected, show the help window and the input frame
            # without applying any filter
            cv.imshow('help',help_window)
            cv.imshow('input',frame)
            continue
        # If the region is valid, apply the filters to the selected portion
        portion = frame[y1:y2,x1:x2]
        cv.rectangle(frame, (x1,y1), (x2,y2), color=(0,255,255), thickness=2)
    else: portion =  frame

    # if a key is pressed, apply the corresponding filter
    if (key in change_keys):
        filter_selection(key)

    #If at least one filter should be applied
    if (not list_applied[0]):
        #Convert the portion we want to apply the filter to grayscale
        portion = cv.cvtColor(portion,cv.COLOR_BGR2GRAY)
        #BoxFilter
        if list_applied[1]:
            boxSize = box_value[0]
            if boxSize > 0: # If the box size is 0, the filter will crash
                portion = cv.boxFilter(portion,-1,(boxSize,boxSize))
                putText(frame,f"1. boxVariable = {box_value[0]}", orig = (5,35))
            # else filter not applied

        #GaussianBlur
        if list_applied[2]:
            gaussSize = gauss_value[0]
            if gaussSize > 0: # If the gaussian sigma value is 0, the filter will crash
                portion = cv.GaussianBlur(portion,(0,0),gaussSize)
                putText(frame,f"2. gaussianVariable = {gauss_value[0]}", orig = (5,55))
            # else filter not applied

        #MedianBlur
        if list_applied[3]:
            medSize = median_value[0]
            medSize = int(medSize) if (medSize%2) else int(medSize)+1
            portion = cv.medianBlur(portion,medSize)
            putText(frame,f"3. medianVariable = {median_value[0]}", orig = (5,75))

        #BilateralFilter
        if list_applied[4]:
            bilSize = bilateral_value[0]
            bilSize = int(bilSize) if (bilSize%2) else int(bilSize)+1
            portion = cv.bilateralFilter(portion,bilSize,75,75)
            putText(frame,f"4. bilateralVariable = {bilateral_value[0]}", orig = (5,95))

        #MinimumFilter
        if list_applied[5]:
            minSize = minimum_value[0]
            minSize = int(minSize) if (minSize%2) else int(minSize)+1
            portion = minimum_filter(portion, minSize)
            putText(frame,f"5. minimumVariable = {minimum_value[0]}", orig = (5,115))

        #MaximumFilter
        if list_applied[6]:
            maxSize = maximum_value[0]
            maxSize = int(maxSize) if (maxSize%2) else int(maxSize)+1
            portion = maximum_filter(portion, maxSize)
            putText(frame,f"6. maximumVariable = {maximum_value[0]}", orig = (5,135))


        #If filters were applied to a portion of the frame, replace it with the filtered portion
        if apply_to_portion:
            frame[y1:y2,x1:x2] = cv.merge([portion,portion,portion])
            
        #Otherwise, replace the entire frame with the filtered portion
        else:
            frame = portion
    
    #Show the resulting frame
    cv.imshow('input',frame)
    #If the help window is open, show it
    if show_help: cv.imshow('help',help_window)
    #If the 'q' key is pressed, exit the loop
    if key == ord('q'): break

#When finished, close all windows
cv.destroyAllWindows()
