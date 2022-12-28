#Matthew Leung
#February 15, 2022
"""
This script is just a test to acquire video (still frames), and then apply the
functions in Gaussian_Fit.py to frames. In a loop, keeps taking camera frames
using OpenCV VideoCapture, and then for each frame:
    1) Iteratively find centroid of the frame/image (find_centroid_iter)
    2) Crop the image about the centroid (crop_image_for_fit)
    3) Fit a 2D Gaussian function to the cropped image (fit_gaussian)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2 as cv
from Gaussian_Fit import find_centroid_iter, crop_image_for_fit, fit_gaussian, gaussian

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
    
        img = 255 - cv.cvtColor(frame, cv.COLOR_BGR2GRAY) #invert image

        x_cen, y_cen = find_centroid_iter(img, cen_tol=5, maxiter=10, do_plot=False)
        cropped_img, crop_radius = crop_image_for_fit(img, x_cen, y_cen, radius_scale=3)
        cv.rectangle(img,(int(x_cen-crop_radius),int(y_cen+crop_radius)),(int(x_cen+crop_radius),int(y_cen-crop_radius)),(255,0,0),2)
        cv.circle(img,(int(x_cen),int(y_cen)), 5, (0,0,255), 1)
        #params, params_uncert, params_dict, params_uncert_dict, fit_fcn = fit_gaussian(cropped_img)
        #x = np.arange(cropped_img.shape[1])
        #y = np.arange(cropped_img.shape[0])
        #X, Y = np.meshgrid(x, y)
        #Z = gaussian(X, Y, *params)

        cv.imshow('img',img)
        #cv.imshow('cimg', cropped_img)
        
        k = cv.waitKey(5) & 0xFF
        if k == 27: #esc key pressed
            break

    cap.release()
    cv.destroyAllWindows()

