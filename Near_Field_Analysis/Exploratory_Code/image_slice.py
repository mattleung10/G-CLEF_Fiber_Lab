#Matthew Leung
#April 2022

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import PIL.Image as Image
import warnings

def read_12bit_image_saved_as_16bit(filename, verbosity=0):
    """
    Reads a 12 bit image which was saved as a 16 bit image
    """
    img16bit = Image.open(filename)
    arr16bit = np.asarray(img16bit)
    arr12bit = np.divide(arr16bit, 2**4)
    if verbosity >= 1:
        num_unique_elems = np.size(np.unique(arr12bit))
        print("Image has {} unique elements".format(num_unique_elems))
        print("Image maximum value is {}".format(np.max(arr12bit)))
        print("Image minimum value is {}".format(np.min(arr12bit)))
    return arr12bit

def otsu_method(img, bit_depth=8):
    """
    Otsu's Method, modified from
    https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
    to allow for bit depths other than just 8 bit
    """
    blur = img
    tot_bins = int(2**bit_depth)
    
    # find normalized_histogram, and its cumulative distribution function
    hist = cv.calcHist([blur],[0],None,[tot_bins],[0,tot_bins])
    hist_norm = hist.ravel()/hist.sum()
    Q = hist_norm.cumsum()
    bins = np.arange(tot_bins)
    fn_min = np.inf
    thresh = -1
    for i in range(1,tot_bins):
        p1,p2 = np.hsplit(hist_norm,[i]) # probabilities
        q1,q2 = Q[i],Q[tot_bins-1]-Q[i] # cum sum of classes
        if q1 < 1e-6 or q2 < 1e-6:
            continue
        b1,b2 = np.hsplit(bins,[i]) # weights
        # finding means and variances
        m1,m2 = np.sum(p1*b1)/q1, np.sum(p2*b2)/q2
        v1,v2 = np.sum(((b1-m1)**2)*p1)/q1,np.sum(((b2-m2)**2)*p2)/q2
        # calculates the minimization function
        fn = v1*q1 + v2*q2
        if fn < fn_min:
            fn_min = fn
            thresh = i
    return thresh


def find_edge_contours(img, bit_depth=8, use_cv_otsu=False, verbosity=0):
    """
    Fits an ellipse to the image. Finds the largest ellipse.
    Returns None if there's an error.
    Steps:
        1) Apply Gaussian blurring to smooth out the image
        2) Apply Otsu's Method to threshold the smoothed image
        3) Find contours in thresholded image
        4) For each contour, fit an ellipse to it
        5) Find the largest ellipse
    INPUTS:
        ::np.ndarray:: img      #Input image
        ::int:: bit_depth       #Bit depth (e.g. 8 or 12)
        ::boolean:: use_cv_otsu #Whether or not to use OpenCV's Otsu Binarization
        ::int:: verbosity       #Amount of details to plot/print
    OUTPUTS:
        ::list of np.ndarray:: valid_contours
        ::list of tuple:: valid_ellipses
        ::list of list:: processed_ellipses
        ::int:: max_area_index
    """
    #Some resources:
    #https://docs.opencv.org/3.4/d4/d73/tutorial_py_contours_begin.html
    #https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contour_features/py_contour_features.html
    #https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
    #https://stackoverflow.com/questions/38530899/python-fit-ellipse-to-an-image
    
    #Gaussian blur the image to smooth out noise
    blurred_img = cv.GaussianBlur(img, (5, 5), 0)
    if verbosity >= 3: #Plot histogram of blurred_img
        plt.figure()
        plt.imshow(blurred_img)
        plt.title('Blurred image')
        plt.show()
        
        plt.figure()
        plt.hist(blurred_img.ravel(), 256)
        plt.title('Histogram of Gaussian blurred image')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.show()
    
    #Threshold the image using Otsu's Method
    #https://en.wikipedia.org/wiki/Otsu%27s_method
    if use_cv_otsu == True and bit_depth == 8:
        #Use the OpenCV implementation, which only works for 8 bit
        _, thresh = cv.threshold(blurred_img, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    else:
        otsu_thresh_val = otsu_method(img, bit_depth=bit_depth)
        #_, thresh = cv.threshold(blurred_img, otsu_thresh_val, bit_depth**2-1, cv.THRESH_BINARY)
        thresh = np.where(thresh >= otsu_thresh_val, bit_depth**2-1, 0)
        if verbosity >= 3: print("Threshold from Otsu's Method is {}".format(otsu_thresh_val))
    if verbosity >= 3:
        plt.figure()
        plt.imshow(thresh)
        plt.title("Thresholded blurred image using Otsu's Method")
        plt.show()
    
    #If needed, convert binary image to 8 bit
    if bit_depth != 8:
        thresh = (thresh / (bit_depth**2-1) * (2**8-1)).astype('uint8')
    
    #Fill the thresholded image, and then apply contours afterwards
    #Don't do this because it's actually slower
    #filled_binary_img = scipy.ndimage.morphology.binary_fill_holes(thresh)
    #thresh = filled_binary_img.astype(np.uint8) * 255
    
    #Find contours of thresholded image
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        warnings.warn("No contours were found")
        return None
    return contours, hierarchy


if __name__ == "__main__":
    filename = 'Images/20220413_141608.tif'
    filename = 'Images/20220413_141650.tif'
    #filename = 'Images/20220413_143326.tif'
    img = read_12bit_image_saved_as_16bit(filename)
    
    vert_px = 320
    vert_px = 320
    
    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.axhline(vert_px, c='r')
    plt.show()
    
    xmin = 225
    xmax = 400
    ymin = 800
    ymax = 975
    roi = img[xmin:xmax, ymin:ymax]
    plt.figure()
    plt.imshow(roi, cmap='gray')
    plt.axhline(vert_px-xmin, c='r')
    plt.show()
    
    hor_slice = roi[vert_px-xmin,:]
    plotpoints = np.arange(np.size(hor_slice))
    plt.plot(plotpoints, hor_slice, c='r')
    plt.ylabel('Pixel Value')
    plt.show()
    
    ###########################################################################
    thresh = 1000
    thresholded = np.where(hor_slice > thresh, hor_slice, 0)
    above = thresholded[np.where(thresholded != 0)]
    mean_of_above = np.mean(above)
    std_of_above = np.std(above)
    
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    fig, ax = plt.subplots(figsize=(8,10), dpi=100)
    ax.xaxis.set_ticks_position('top')
    #ax.xaxis.set_label_position('top')
    ax.tick_params(left=True, bottom=True, top=True)
    divider = make_axes_locatable(ax)
    bottom1_ax = divider.append_axes("bottom", size=3, pad=0.1, sharex=ax)
    #Note: size and pad should be axes_grid.axes_size compatible
    bottom1_ax.xaxis.set_tick_params(labelbottom=True) #show xlabel
    bottom1_ax.tick_params(left=True, bottom=True, top=False)
    
    #ax.set_xlabel('Horizontal Pixel Number')
    ax.set_ylabel('Vertical Pixel Number')
    ax.imshow(roi, cmap='gray', aspect='equal')
    ax.axhline(vert_px-xmin, color='r')
    
    ax.autoscale(enable=False)
    bottom1_ax.autoscale(enable=False)
    bottom1_ax.set_ylim(top=2**12 * 1.1)
    bottom1_ax.set_ylabel('Pixel Value')
    bottom1_ax.set_xlabel('Horizontal Pixel Number')
    
    plot_text = '$\mu/\sigma$ = {:.3f}'.format(mean_of_above/std_of_above)
    bottom1_ax.plot(plotpoints, hor_slice, c='r', label='Slice') #plot the slice
    bottom1_ax.axhline(mean_of_above, color='blue', linestyle='--', label='Mean $\mu$')
    bottom1_ax.axhspan(mean_of_above+std_of_above, mean_of_above-std_of_above, color='blue', alpha=0.1, label='$1\sigma$ from Mean $\mu$')
    bottom1_ax.legend(loc='lower center')
    bottom1_ax.text(x=0.02,y=0.9, s=plot_text, fontsize=18, transform=bottom1_ax.transAxes)
    plt.savefig(os.path.basename(filename)[:-4]+'_slice.png', bbox_inches='tight')
    plt.show()
    
    print("mean over std = {}".format(mean_of_above/std_of_above))
    
    
    
    ###########################################################################
    
    #contours, hierarchy = find_edge_contours(roi, bit_depth=12, use_cv_otsu=True, verbosity=3)