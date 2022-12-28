#Matthew Leung
#Code last modified: March 16, 2022
"""
This script is used to focus the arms containing the Matrix Vision cameras.
Uses Harvester Python library to control camera and to acquire images.
Uses matplotlib interactive mode to display results, like what was done for the
Pixelink cameras (hence the "from_pixelink" in the filename).

For each image, using the functions in Gaussian_Fit.py, this script:
    1) Iteratively finds centroid of the frame/image (find_centroid_iter)
    2) Crops the image about the centroid (crop_image_for_fit)
    3) Fits a 2D Gaussian function to the cropped image (fit_gaussian)
    4) Finds the average FWHM of the fitted Gaussian
It then plots the results in realtime, and plots time VS average FWHM.

Note that there are several issues:
    1) The method of iteratively finding the centroid can be unreliable if
       the spot is very small compared to the rest of the image
    2) matplotlib interactive mode is not good at displaying live images;
       program can easily lag/crash
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2 as cv
import time
from Gaussian_Fit import find_centroid_iter, crop_image_for_fit, fit_gaussian, gaussian, gaussian_FWHM
from matrixvision_harvester_class import matrixvision_harvester_class as matrixvision_harvester_class

if __name__ == "__main__":
    cti_filename = r"C:\Program Files\MATRIX VISION\mvIMPACT Acquire\bin\x64\mvGenTLProducer.cti"
    camera_obj = matrixvision_harvester_class()
    camera_obj.init_camera(cti_filename)
    camera_obj.start_camera_acquisition(pixel_format='Mono8')
    camera_obj.set_min_exposure_time(10) #set the minimum exposure time on the mvBlueCOUGAR-X102kG
    camera_obj.set_exposure(exp_time=10)
    camera_obj.set_frame_rate(1) #Set to low FPS
        
    def take_frame():
        #take each frame
        raw_image = camera_obj.get_snapshot_np_array()
        camera_obj.buffer.queue()
        del camera_obj.buffer #delete the buffer object
        
        #THIS LINE BELOW IS CRITICAL!
        #raw_image has shape (1104, 1600, 1) but we need an image of shape (1104, 1600)
        return raw_image[:,:,0] 
    
    #Keep taking frames until you get a valid frame. This is done to get rid
    #of the bad frames at the beginning of the acquisition.
    while True:
        img = take_frame()
        if img.size == 0: #check if img is empty
            print("Image empty")
            continue
        if np.max(img) == np.min(img): #check if img has the same values for all pixels
            print("Image has same values for all pixels")
            plt.figure()
            plt.imshow(img)
            plt.show()
            continue
        break

    start_time = time.time() #starting time
    
    x_cen, y_cen = find_centroid_iter(img, cen_tol=5, maxiter=10, do_plot=False) #iteratively find the centroid of the image
    cropped_img, crop_radius = crop_image_for_fit(img, x_cen, y_cen, radius_scale=3) #crop the image
    h, w = cropped_img.shape
    
    #Fit Gaussian
    params, params_uncert, params_dict, params_uncert_dict, fit_fcn = fit_gaussian(cropped_img, clipped_gaussian=True)
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
    tx = ax[2].text(x=0.02,y=0.07, s=plot_text_x, fontsize=24, transform=ax[2].transAxes, color='r')
    ty = ax[2].text(x=0.02,y=-0.05, s=plot_text_y, fontsize=24, transform=ax[2].transAxes, color='limegreen')

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

    plt.ion() #turn on matplotlib interactive mode
    try: #to break the while loop, type Ctrl+C / Ctrl+Z
        while True:
            img = take_frame()
            if img.size == 0: #check if img is empty
                continue
            if np.max(img) == np.min(img): #check if img has the same values for all pixels
                continue
            frame_time = time.time() #time at which frame was taken

            x_cen, y_cen = find_centroid_iter(img, cen_tol=5, maxiter=10, do_plot=False) #iteratively find the centroid of the image
            cropped_img, crop_radius = crop_image_for_fit(img, x_cen, y_cen, radius_scale=1) #crop the image
            if cropped_img.size == 0: #if image is empty, ignore this image and move on
                continue
            if 0 in cropped_img.shape: #if image has an empty dimension, ignore this image and move on
                continue
            h, w = cropped_img.shape
            
            #Fit Gaussian
            params, params_uncert, params_dict, params_uncert_dict, fit_fcn = fit_gaussian(cropped_img, clipped_gaussian=True)
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
    except KeyboardInterrupt: #break the loop
        pass
        
    plt.ioff() #turn off matplotlib interactive mode
    plt.show()
    
    camera_obj.done_camera() #finished using camera; you must do this
    print("Done!")

