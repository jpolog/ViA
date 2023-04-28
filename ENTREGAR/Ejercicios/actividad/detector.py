#!/usr/bin/env python
from datetime import datetime
from collections import deque
import time
import cv2 as cv
from umucv.util import ROI, putText

from umucv.stream import autoStream

# Create a window an reate a ROI object within the "input" window
cv.namedWindow("input")
roi = ROI("input")

# video framerate
fps = 60
change_threshold = 0.01
# store the last frames
listFrames = deque(maxlen=fps*3)


# Create a background subtractor that returns a mask with 0s for background pixels and 255s for foreground pixels
bg_subtractor = cv.createBackgroundSubtractorMOG2(500, 16, False)

# Flag for recording state
recording = False

# Main loop for processing video frames
for key, frame in autoStream():

    # If ROI is defined, process the region of interest
    if roi.roi:
        x1, y1, x2, y2 = roi.roi
        # If the region is not valid, show the input frame
        # without processing it
        if (x2-x1)*(y2-y1) <= 100 or (x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]): 
            cv.imshow('input',frame)
            continue

        # Draw a rectangle around the ROI
        cv.rectangle(frame, (x1, y1), (x2, y2), color=(0, 255, 255), thickness=1)
        putText(frame, f"{x2 - x1 + 1}x{y2 - y1 + 1}", orig=(x1, y1 - 8))

        # Remove the ROI if the 'x' key is pressed
        if key == ord("x"):
            roi.roi = []
            cv.destroyWindow("roi")
            

        # Add the current frame to the list of frames
        # in case movement is detected and 
        # we need to record the frames.
        roi_frame = frame[y1+1:y2, x1+1:x2]
        listFrames.append(roi_frame)

        # Background subtraction:
        # Extract the region of interest from the frame
        cv.imshow("roi", roi_frame)

        #If we have not recorded enough frames, show the input frame
        if len(listFrames) < fps*3:
            cv.imshow("input", frame)
            continue

        # Apply background subtraction to the region of interest and obtain a mask
        foreground_mask = bg_subtractor.apply(roi_frame)
        masked = roi_frame.copy()
        masked[foreground_mask == 0] = 0

        # Calculate the percentage of pixels that have changed
        perc_changed = foreground_mask.sum() / foreground_mask.size

        # Record if the percentage of changed pixels is greater than change_threshold
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

        # Continue recording
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

        # Display the masked region of interest
        cv.imshow("Detector", masked)


    # Display the input frame and its dimensions
    h, w, _ = frame.shape
    putText(frame, f"{w}x{h}")
    cv.imshow("input", frame)
    
    if key == ord("q"):
        break

# Clean up all windows
cv.destroyAllWindows()
