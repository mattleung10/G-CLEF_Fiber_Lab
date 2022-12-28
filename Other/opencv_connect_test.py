import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2 as cv

if __name__ == "__main__":
    #https://stackoverflow.com/questions/52043671/opencv-capturing-image-with-black-side-bars/56750151#56750151
    cap = cv.VideoCapture(0, cv.CAP_DSHOW)
    if cap.isOpened() != True:
        print("Camera cannot be opened")
        exit()
    
    while True:
        #take each frame
        ret, frame = cap.read()
        if ret == False:
            break
        if frame.size == 0:
            continue
        if np.max(frame) == np.min(frame):
            continue
    
        img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        cv.imshow('img',img)
        #cv.imshow('cimg', cropped_img)
        
        k = cv.waitKey(5) & 0xFF
        if k == 27: #esc key pressed
            break

    cap.release()
    cv.destroyAllWindows()