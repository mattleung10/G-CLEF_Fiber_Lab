#Matthew Leung
#February 15, 2022
"""
Summary:
Acquires images and applies functions in Gaussian_Fit.py;
OpenCV VideoCapture will be replaced with function from
Pixelink Python API (or whatever camera API you want to use) later on.

Description:
This script uses OpenCV to acquire video. For each frame, the following is done:
    1) Iteratively find centroid of the frame/image (find_centroid_iter)
    2) Crop the image about the centroid (crop_image_for_fit)
    3) Fit a 2D Gaussian function to the cropped image (fit_gaussian)
Then the following are plotted:
    1) Original image/frame
    2) Cropped region
    3) Fitted Gaussian
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2 as cv
from Gaussian_Fit import find_centroid_iter, crop_image_for_fit, fit_gaussian, gaussian, gaussian_FWHM

if __name__ == "__main__":
    cap = cv.VideoCapture(0, cv.CAP_DSHOW)
    if cap.isOpened() != True:
        print("Camera cannot be opened")
        exit()
        
    def take_frame():
        #take each frame
        ret, frame = cap.read()
        return 255 - cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    img = take_frame()
    x_cen, y_cen = find_centroid_iter(img, cen_tol=5, maxiter=10, do_plot=False)
    cropped_img, crop_radius = crop_image_for_fit(img, x_cen, y_cen, radius_scale=3)
    params, params_uncert, params_dict, params_uncert_dict, fit_fcn = fit_gaussian(cropped_img)
    FWHM_x = gaussian_FWHM(params_dict['sigma_x'])
    FWHM_y = gaussian_FWHM(params_dict['sigma_y'])
    x = np.arange(cropped_img.shape[1])
    y = np.arange(cropped_img.shape[0])
    X, Y = np.meshgrid(x, y)
    Z = gaussian(X, Y, *params)

    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(16,4), constrained_layout=True)
    im0 = ax[0].imshow(img, cmap='gray')
    rect = patches.Rectangle((x_cen-crop_radius, y_cen-crop_radius), crop_radius*2, crop_radius*2, linewidth=1, edgecolor='r', facecolor='none')
    ax[0].add_patch(rect)
    ax[0].set_title('Original Image')
    
    im1 = ax[1].imshow(cropped_img, cmap='gray')
    ax[1].set_title('Cropped Region')
    ax[1].get_xaxis().set_visible(False) #Hide ticks
    ax[1].get_yaxis().set_visible(False)
    
    im2 = ax[2].imshow(Z, cmap='gray')
    ax[2].set_title('Gaussian Fit')
    ax[2].get_xaxis().set_visible(False) #Hide ticks
    ax[2].get_yaxis().set_visible(False)
    plot_text_x = "FWHM$_{x}$ = " + "{:.3f} pixels".format(FWHM_x)
    plot_text_y = "FWHM$_{y}$ = " + "{:.3f} pixels".format(FWHM_y)
    tx = ax[2].text(x=0.02,y=0.07, s=plot_text_x, fontsize=12, transform=ax[2].transAxes, color='r')
    ty = ax[2].text(x=0.02,y=0.00, s=plot_text_y, fontsize=12, transform=ax[2].transAxes, color='limegreen')

    h, w = cropped_img.shape
    x_slice, = ax[3].plot(np.arange(w),np.zeros(w), 'r-')
    ax[3].set_title('Horizontal Slice of Cropped Region')

    plt.ion()
    try:
        while True:
            img = take_frame()
            if img.size == 0: #check if img is empty
                continue
            if np.max(img) == np.min(img): #check if img has the same values for all pixels
                continue
            
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
            x = np.arange(cropped_img.shape[1])
            y = np.arange(cropped_img.shape[0])
            X, Y = np.meshgrid(x, y)
            Z = gaussian(X, Y, *params)
    
            im0.set_data(img)
            im0.set_clim(np.min(img), np.max(img))
            rect.set_xy([x_cen-crop_radius, y_cen-crop_radius])
            rect.set_height(crop_radius*2)
            rect.set_width(crop_radius*2)
            
            im1.set_data(cropped_img)
            im1.set_clim(np.min(cropped_img), np.max(cropped_img))
            im2.set_data(Z)
            im2.set_clim(np.min(Z), np.max(Z))
            plot_text_x = "FWHM$_{x}$ = " + "{:.3f} pixels\n".format(FWHM_x)
            plot_text_y = "FWHM$_{y}$ = " + "{:.3f} pixels\n".format(FWHM_y)
            tx.set_text(plot_text_x)
            ty.set_text(plot_text_y)
            
            x_slice.set_data(np.arange(w), cropped_img[int(crop_radius),:])
            ax[3].set_ylim([0,np.max(cropped_img)*1.1])
            ax[3].set_xlim([0,w])
            
            plt.pause(0.05)
    except KeyboardInterrupt:
        pass
        
    plt.ioff()
    plt.show()
    
    cap.release()
    print("Done!")

