#!/usr/bin/env python

from datetime import datetime
import cv2 as cv
import numpy as np
from umucv.util import ROI, putText
from umucv.stream import autoStream
from collections import deque


# Create a window an reate a ROI object within the "input" window
cv.namedWindow("input")
roi = ROI("input")

# video framerate
fps = 60
change_threshold = 0.02
# store the last frames
listFrames = deque(maxlen=fps*3)
listFrames_gray = deque(maxlen=fps*3)

recording = False
roi_dimen = [0,0,0,0]


for key, frame in autoStream():

        
    #If the ROI is defined and we have already recorded some frames
    if roi.roi:
        #Corners of the roi of interest
        x1, y1, x2, y2 = roi.roi
        # If the region is not valid, show the input frame
        # without processing it
        if (x2-x1)*(y2-y1) <= 100 or (x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]): 
            cv.imshow('input',frame)
            continue
        # todos los frames del buffer tienen que tener la misma dimension
        # si cambia el roi se vacía el buffer
        if roi_dimen != roi.roi:
            roi_dimen = roi.roi
            listFrames.clear()
            listFrames_gray.clear()
            cv.imshow("input", frame)
            continue

        #Draw the rectangle that forms the ROI
        cv.rectangle(frame, (x1, y1), (x2, y2), color=(0, 255, 255), thickness=1)
        putText(frame, f'{x2 - x1}x{y2 - y1}', orig=(x1, y1 - 8))

        #Remove the ROI if the 'x' key is pressed
        if key == ord('x'):
            roi.roi = []
            cv.destroyWindow("old_roi_frame")
            cv.destroyWindow("roi_frame")
            cv.destroyWindow("masked")

        # Extract the region of interest from the frame
        roi_frame = frame[y1+1:y2, x1+1:x2]
        listFrames.append(roi_frame)

        roi_frame_g = cv.cvtColor(roi_frame, cv.COLOR_BGR2GRAY)
        listFrames_gray.append(roi_frame_g)

        #If we have not recorded enough frames, show the input frame
        if len(listFrames) < fps*3:
            cv.imshow("input", frame)
            continue

        #Extract the region of interest from the last frames
        # and calculatte the absolute difference between the two regions
        # in multiple frames
        old_roi_frame0 = listFrames_gray[fps*3-5]
        old_roi_frame1 = listFrames_gray[fps*3-4]
        old_roi_frame2 = listFrames_gray[fps*3-3]
        old_roi_frame3 = listFrames_gray[fps*3-2]
        dif0 = cv.absdiff(roi_frame_g, old_roi_frame0)
        dif1 = cv.absdiff(roi_frame_g, old_roi_frame1)
        dif2 = cv.absdiff(roi_frame_g, old_roi_frame2)
        dif3 = cv.absdiff(roi_frame_g, old_roi_frame3)
    
        #Create a mask with the pixels that have changed wheigthed by the frame
        # in which they have changed (we want to give more importance to the 
        # pixels that have changed in the older frames, because they are more
        # likely to be the background)
        mask = dif0*0.35 +  dif1*0.25 + dif2*0.20 + dif3*0.20 > 10

        # Apply the mask to the region of interest
        trozo_masked = cv.bitwise_and(roi_frame_g, roi_frame_g, mask=mask.astype(np.uint8))
 
    
        # if the region masked is bigger than 10% of the ROI, record the video
        perc_changed = np.sum(mask)/mask.size
        if perc_changed > change_threshold and not recording:
            recording = True
            frames_recorded = 0 # number of frames written to the video
            print("Recording")
            # Create a video object 
            # Define the frame rate and video size
            width = x2 - x1-1
            height = y2 - y1-1

            # Create a VideoWriter object
            fourcc = cv.VideoWriter_fourcc(*'mp4v')  # Choose the codec (e.g. 'XVID', 'MJPG', 'mp4v')
            fname = datetime.now().strftime("%Y%m%d-%H%M%S.") + "mp4"
            out = cv.VideoWriter(fname, fourcc, fps, (width, height))
    

        if (recording == True):
            frames_recorded += 1
            # Stop recording if 2 more seconds = 2*fps frames passed and 
            # the frame has stopped changing
            if (frames_recorded >= 2*fps):
                # write all frames in listFrames to the video
                for frame in listFrames:
                    out.write(frame)

                if perc_changed > change_threshold:
                    # keep recording if the region keeps changing
                    # record another 3 seconds = 3*fps 
                    frames_recorded = -fps
                else:
                    out.release()
                    recording = False
                    print("Stopped recording")


        #Show the images
        cv.imshow("roi_frame", roi_frame)
        cv.imshow("old_roi_frame", old_roi_frame0)
        cv.imshow("masked", trozo_masked)




    #Return the height and width of the frame
    h, w, _ = frame.shape
    putText(frame, f'{w}x{h}')
    cv.imshow('input', frame)

    if key == ord("q"):
        break

cv.destroyAllWindows()
