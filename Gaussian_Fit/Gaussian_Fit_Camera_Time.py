#Matthew Leung
#February 24, 2022
"""
Summary:
Acquires images and applies functions in Gaussian_Fit.py, and also finds the
average FWHM of the fitted Gaussian; OpenCV VideoCapture will be replaced with function
from Pixelink Python API (or whatever camera API you want to use) later on.
THIS IS BASICALLY THE SAME AS Gaussian_Fit_Camera.py, EXCEPT THAT A PLOT OF
TIME VS AVERAGE FWHM IS MADE.

Description:
This script uses OpenCV to acquire video. For each frame, the following is done:
    1) Iteratively find centroid of the frame/image (find_centroid_iter)
    2) Crop the image about the centroid (crop_image_for_fit)
    3) Fit a 2D Gaussian function to the cropped image (fit_gaussian)
Then the following are plotted:
    1) Original image/frame
    2) Cropped region
    3) Fitted Gaussian
    4) Plot of time VS average FWHM
This script has "Time" in its filename because it will display a plot of time
VS average FWHM (where average FWHM is that of the fitted Gaussian function)
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2 as cv
import time
from Gaussian_Fit import find_centroid_iter, crop_image_for_fit, fit_gaussian, gaussian, gaussian_FWHM

if __name__ == "__main__":
    cap = cv.VideoCapture(0, cv.CAP_DSHOW)
    if cap.isOpened() != True:
        print("Camera cannot be opened")
        exit()
        
    def take_frame():
        #take each frame
        ret, frame = cap.read()
        return 255 - cv.cvtColor(frame, cv.COLOR_BGR2GRAY) #invert image
    
    img = take_frame()
    start_time = time.time() #starting time
    
    x_cen, y_cen = find_centroid_iter(img, cen_tol=5, maxiter=10, do_plot=False)
    cropped_img, crop_radius = crop_image_for_fit(img, x_cen, y_cen, radius_scale=3)
    h, w = cropped_img.shape
    
    #Fit Gaussian
    params, params_uncert, params_dict, params_uncert_dict, fit_fcn = fit_gaussian(cropped_img)
    FWHM_x = gaussian_FWHM(params_dict['sigma_x'])
    FWHM_y = gaussian_FWHM(params_dict['sigma_y'])
    FWHM_avg = (FWHM_x + FWHM_y)/2
    x = np.arange(cropped_img.shape[1])
    y = np.arange(cropped_img.shape[0])
    X, Y = np.meshgrid(x, y)
    Z = gaussian(X, Y, *params)

    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(16,4), constrained_layout=True)
    
    #Plot the original image
    im0 = ax[0].imshow(img, cmap='gray')
    rect = patches.Rectangle((x_cen-crop_radius, y_cen-crop_radius), crop_radius*2, crop_radius*2, linewidth=1, edgecolor='r', facecolor='none')
    ax[0].add_patch(rect)
    ax[0].set_title('Original Image')
    
    #Plot the cropped region
    im1 = ax[1].imshow(cropped_img, cmap='gray')
    ax[1].set_title('Cropped Region')
    ax[1].get_xaxis().set_visible(False) #Hide ticks
    ax[1].get_yaxis().set_visible(False)
    
    #Plot the fitted Gaussian
    im2 = ax[2].imshow(Z, cmap='gray')
    ax[2].set_title('Gaussian Fit')
    ax[2].get_xaxis().set_visible(False) #Hide ticks
    ax[2].get_yaxis().set_visible(False)
    plot_text_x = "FWHM$_{x}$ = " + "{:.3f} pixels".format(FWHM_x)
    plot_text_y = "FWHM$_{y}$ = " + "{:.3f} pixels".format(FWHM_y)
    tx = ax[2].text(x=0.02,y=0.07, s=plot_text_x, fontsize=12, transform=ax[2].transAxes, color='r')
    ty = ax[2].text(x=0.02,y=0.00, s=plot_text_y, fontsize=12, transform=ax[2].transAxes, color='limegreen')

    #Plot time VS average FWHM
    prepoints = 50 #number of points before start of data acquisition
    time_array = np.linspace(-10,0,num=prepoints)
    FWHM_array = np.zeros(prepoints)
    time_plot, = ax[3].plot(time_array,FWHM_array, 'r-')
    ax[3].set_title('Average FWHM VS Time')
    ax[3].set_xlabel('Time [seconds]')
    ax[3].set_ylabel('Average FWHM [pixels]')
    ax[3].set_xlim([np.min(time_array), np.max(time_array)])
    ax[3].set_ylim([0, np.max(FWHM_avg)*1.1])

    plt.ion()
    try:
        while True:
            img = take_frame()
            if img.size == 0: #check if img is empty
                continue
            if np.max(img) == np.min(img): #check if img has the same values for all pixels
                continue
            frame_time = time.time() #time at which frame was taken
            
            x_cen, y_cen = find_centroid_iter(img, cen_tol=5, maxiter=10, do_plot=False)
            cropped_img, crop_radius = crop_image_for_fit(img, x_cen, y_cen, radius_scale=3)
            if cropped_img.size == 0:
                continue
            if 0 in cropped_img.shape:
                continue
            h, w = cropped_img.shape
            
            params, params_uncert, params_dict, params_uncert_dict, fit_fcn = fit_gaussian(cropped_img)
            FWHM_x = gaussian_FWHM(params_dict['sigma_x'])
            FWHM_y = gaussian_FWHM(params_dict['sigma_y'])
            FWHM_avg = (FWHM_x + FWHM_y)/2
            x = np.arange(cropped_img.shape[1])
            y = np.arange(cropped_img.shape[0])
            X, Y = np.meshgrid(x, y)
            Z = gaussian(X, Y, *params)
    
            #Update original image
            im0.set_data(img)
            im0.set_clim(np.min(img), np.max(img))
            rect.set_xy([x_cen-crop_radius, y_cen-crop_radius])
            rect.set_height(crop_radius*2)
            rect.set_width(crop_radius*2)
            
            #Update cropped region
            im1.set_data(cropped_img)
            im1.set_clim(np.min(cropped_img), np.max(cropped_img))
            
            #Update fitted Gaussian
            im2.set_data(Z)
            im2.set_clim(np.min(Z), np.max(Z))
            plot_text_x = "FWHM$_{x}$ = " + "{:.3f} pixels\n".format(FWHM_x)
            plot_text_y = "FWHM$_{y}$ = " + "{:.3f} pixels\n".format(FWHM_y)
            tx.set_text(plot_text_x)
            ty.set_text(plot_text_y)
            
            #Update time_array
            time_array = np.append(time_array, frame_time - start_time)            
            time_array = time_array[1:]
            #Update FWHM array
            FWHM_array = np.append(FWHM_array, FWHM_avg)
            FWHM_array = FWHM_array[1:]
            #Update Time VS Average FWHM plot
            time_plot.set_data(time_array, FWHM_array)
            ax[3].set_xlim([np.min(time_array), np.max(time_array)])
            ax[3].set_ylim([0, np.max(FWHM_array*1.1)])
            
            plt.pause(0.05)
    except KeyboardInterrupt:
        pass
        
    plt.ioff()
    plt.show()
    
    cap.release()
    print("Done!")

