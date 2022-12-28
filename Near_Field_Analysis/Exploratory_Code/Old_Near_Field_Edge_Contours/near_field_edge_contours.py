#Matthew Leung
#Code last updated: March 14, 2022
"""
This script finds the contour bounding the face of a circular fiber,
(in a near field image) and fits an ellipse to it. This script was designed for
circular fibers.

Can also be used for a far field image, but the resulting contour will
not enclose the entire shape since the far field image does not have sharp
edges. But this script can provide a good measure of eccentricity of the
image in the far field.

Procedure:
    1) Gaussian blur the image to smooth out noise
    2) Threshold the image using Otsu's Method
    3) Find contours of thresholded image
    4) Fit an ellipse to each contour
    5) Find the eccentricity and area of each ellipse
    6) Find the ellipse with the largest area, and take that as the ellipse
       which encloses the circular fiber face
       
THIS PROCEDURE IS OLD AND ONLY WORKS WELL FOR NEAR FIELD CIRCULAR FIBERS.
FOR FAR FIELD, ONLY USE IT TO DETERMINE THE ECCENTRICITY, AND NOT TO FIND THE
CONTOUR BOUNDING THE FIBER IMAGE.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2 as cv
import scipy.ndimage
import warnings
import time

def find_time_taken(fcn):
    def dummy(*args, **kwargs):
        st = time.time()
        r = fcn(*args, **kwargs)
        print("Function " + fcn.__name__ + " took {} seconds".format(time.time()-st))
        return r
    return dummy


#######################################################################################

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

#######################################################################################

def verify_ellipse_fit_integrity(ellipse):
    """
    Verify if ellipse fit makes sense. Return False if any of the fitted
    parameters are NaN or +/- infinity. Else return True.
    """
    center, axes, angle = ellipse
    if np.any(np.isnan(center)) or np.any(np.isnan(axes)) or np.isnan(angle):
        return False #there are NaN values
    if np.any(np.isinf(center)) or np.any(np.isinf(axes)) or np.isinf(angle):
        return False #there are np.inf values
    return True

def process_contours_ellipse_output(ellipse):
    """
    This function unpacks the values returned by cv.fitEllipse, and also
    calculates the area and eccentricity of the fitted ellipse
    Ellipse equation: (x-x0)^2/a^2 + (y-y0)^2/b^2 = 1
    INPUTS:
        ::tuple:: ellipse
    """
    center, axes, angle = ellipse
    """
    ellipse is a tuple: (center, axes, angle)
        ::tuple:: center #Center of the ellipse.
        ::tuple:: axes	 #Half of the size of the ellipse main axes. [not true, it's not half]
        ::float:: angle	 #Ellipse rotation angle in degrees.
    """
    x0 = center[0] #center x value
    y0 = center[1] #center y value
    a_double = axes[0] #twice the value of a
    b_double = axes[1] #twice the value of b
    
    if a_double <= 0 or b_double <= 0:
        warnings.warn("a = 0 or b = 0")
        return None
    
    area = np.pi * a_double/2 * b_double/2 #area of the ellipse
    
    #Find the eccentricity^2 of the ellipse
    if a_double > b_double:
        eccentricity2 = 1 - b_double**2/a_double**2
    else:
        eccentricity2 = 1 - a_double**2/b_double**2
    
    return x0, y0, a_double, b_double, angle, area, eccentricity2

def find_edge_contours_ellipses(img, bit_depth=8, use_cv_otsu=False, verbosity=0):
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
        _, thresh = cv.threshold(blurred_img, otsu_thresh_val, bit_depth**2-1, cv.THRESH_BINARY)
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
    
    ellipses = [] #list which stores the parameters of the fitted ellipses
    fitted_contours = []
    for cnt_points in contours:
        #cnt_points is a np.ndarray consisting of points in a contour
        if cnt_points.shape[0] >= 5: #Must have at least 5 points to fit ellipse
            ellipse = cv.fitEllipse(cnt_points) #fit ellipse to contour
            """
            ellipse is a tuple: (center, axes, angle)
                ::tuple:: center #Center of the ellipse.
                ::tuple:: axes	 #Half of the size of the ellipse main axes. [not true, it's not half]
                ::float:: angle	 #Ellipse rotation angle in degrees.
            """
            if verify_ellipse_fit_integrity(ellipse) == True:
                ellipses += [ellipse]
            fitted_contours += [cnt_points]
    if len(ellipses) == 0:
        warnings.warn("There were no valid fitted ellipses")
        return None
    
    processed_ellipses = [] #list of ellipse parameters
    valid_contours = []
    valid_ellipses = []
    for i in range(0,len(ellipses),1):
        ellipse = ellipses[i]
        r = process_contours_ellipse_output(ellipse)
        if r is not None: #valid ellipse parameters
            processed_ellipses += [r]
            valid_contours += [fitted_contours[i]]
            valid_ellipses += [ellipse]
    processed_ellipses_np = np.array(processed_ellipses)
    max_positions = np.argmax(processed_ellipses_np, axis=0)
    max_area_index = max_positions[5] #recall area column has index of 5
    largest_ellipse = valid_ellipses[max_area_index] 
    
    if verbosity >= 2:
        plt.figure()
        plt.imshow(img)
        for vals in processed_ellipses:
            x0, y0, a_double, b_double, angle, area, eccentricity2 = vals
            ell = patches.Ellipse((x0, y0), a_double, b_double, angle, linewidth=1, edgecolor='r', facecolor='none')
            plt.gca().add_patch(ell)
        plt.title('All fitted ellipses')
        plt.show()
    
    if verbosity >= 1:
        x0, y0, a_double, b_double, angle, area, eccentricity2 = largest_ellipse
        plt.figure()
        plt.imshow(img)
        ell = patches.Ellipse((x0, y0), a_double, b_double, angle, linewidth=1, edgecolor='r', facecolor='none')
        plt.gca().add_patch(ell)
        plt.title('Largest fitted ellipse')
        plt.show()
        
    return valid_contours, valid_ellipses, processed_ellipses, max_area_index

#######################################################################################

def find_ellipse_major_axis_lines_points(x0, y0, a_double, b_double, angle):
    """
    Finds the endpoints of a line which represents the ellipse major axis
    Modified from: https://stackoverflow.com/a/62701632
    """
    rmajor = max(a_double,b_double)/2
    if angle > 90:
        nangle = angle - 90
    else:
        nangle = angle + 90
    xtop = x0 + np.cos(np.radians(nangle))*rmajor
    ytop = y0 + np.sin(np.radians(nangle))*rmajor
    xbot = x0 + np.cos(np.radians(nangle+180))*rmajor
    ybot = y0 + np.sin(np.radians(nangle+180))*rmajor
    return xtop, ytop, xbot, ybot

def find_ellipse_minor_axis_lines_points(x0, y0, a_double, b_double, angle):
    """
    Finds the endpoints of a line which represents the ellipse minor axis
    Modified from: https://stackoverflow.com/a/62701632
    """
    rminor = min(a_double,b_double)/2
    if angle > 90:
        nangle = angle - 180
    else:
        nangle = angle
    xtop = x0 + np.cos(np.radians(nangle))*rminor 
    ytop = y0 + np.sin(np.radians(nangle))*rminor 
    xbot = x0 + np.cos(np.radians(nangle+180))*rminor 
    ybot = y0 + np.sin(np.radians(nangle+180))*rminor 
    return xtop, ytop, xbot, ybot

def plot_ellipse(img, largest_ellipse, plot_orig_img=False):
    """
    Plot largest_ellipse over img
    INPUTS:
        ::np.ndarray:: img
        ::list:: largest_ellipse
        ::boolean:: plot_orig_img
    """
    if plot_orig_img == True: #plot the original image alone
        plt.figure(dpi=150)
        plt.imshow(img, cmap='gray')
        plt.show()
    
    x0, y0, a_double, b_double, angle, area, eccentricity2 = largest_ellipse
    
    plt.figure(dpi=150)
    plt.imshow(img, cmap='gray')
    plt.scatter(x0, y0, marker='o', s=20, color='r') #show center of ellipse
    
    #Plot the ellipse
    ell = patches.Ellipse((x0, y0), a_double, b_double, angle, linewidth=1, edgecolor='r', facecolor='none')
    ax = plt.gca()
    ax.add_patch(ell)
    
    #Display eccentricity
    plot_text = "$e$ = {:.5f}".format(np.sqrt(eccentricity2))
    
    #Plot major axis
    xtop, ytop, xbot, ybot = find_ellipse_major_axis_lines_points(x0, y0, a_double, b_double, angle)
    plt.plot([int(xtop), int(xbot)], [int(ytop), int(ybot)], color='r', linestyle='--', linewidth=1)
    
    #Plot minor axis
    xtop, ytop, xbot, ybot = find_ellipse_minor_axis_lines_points(x0, y0, a_double, b_double, angle)
    plt.plot([int(xtop), int(xbot)], [int(ytop), int(ybot)], color='r', linestyle='--', linewidth=1)
    
    ax.text(x=0.02,y=0.07, s=plot_text, fontsize=16, transform=ax.transAxes, color='r')
        
    plt.show()
    return True
    
if __name__ == "__main__":
    #filename = 'Images/220301_1634 NF of 200um Fiber, Laser Source.jpg'
    filename = 'Images/220301_1635 NF of 200um Fiber, Laser Source.jpg'
    #filename = 'Images/220301_1626 NF of 200um Fiber, Laser Source.jpg'
    filename = 'Images/20220310 1128 Far Field.png'
    #filename = 'Images/20220310 1130 Far Field No ND.png'
    
    if os.path.isfile(filename) == False:
        raise ValueError("Invalid filename")
    
    img = cv.imread(filename, cv.IMREAD_GRAYSCALE)
    noise = np.round(np.random.normal(25,5,img.shape).reshape(img.shape))
    noise = noise.astype('uint8')
    #img += noise
    
    valid_contours, valid_ellipses, processed_ellipses, max_area_index = find_edge_contours_ellipses(img, verbosity=0)
    largest_processed_ellipse = processed_ellipses[max_area_index]
    plot_ellipse(img, largest_processed_ellipse, plot_orig_img=True)
    
