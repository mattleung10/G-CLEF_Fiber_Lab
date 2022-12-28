#Matthew Leung
#April 2022
"""
A script for preliminary/exploratory analysis of different methods to identify
the face of the fiber
"""

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
    hist = cv.calcHist(blur,[0],None,histSize=[tot_bins],ranges=[0,tot_bins])
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

def apply_sobel_op(img):
    """
    Apply Sobel operator to image
    """
    #Resources:
    #https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_gradients/py_gradients.html
    #https://docs.opencv.org/3.4/d2/d2c/tutorial_sobel_derivatives.html
    #https://docs.opencv.org/3.4/d4/d86/group__imgproc__filter.html#gacea54f142e81b6758cb6f375ce782c8d
    
    #Set depth of output image
    #https://docs.opencv.org/3.4/d4/d86/group__imgproc__filter.html#filter_depths
    #ddepth = cv.CV_16S
    ddepth = cv.CV_64F
    ksize = 3 #kernel size
    
    grad_x = cv.Sobel(img, ddepth, dx=1, dy=0, ksize=ksize) #x gradient
    grad_y = cv.Sobel(img, ddepth, dx=0, dy=1, ksize=ksize) #y gradient
    grad = (np.abs(grad_x) + np.abs(grad_y))/2 #approximation of total gradient
    return grad

def get_image_centroid_vectorized(img, calculate_std=False):
    """
    Fast version of get_image_centroid, with vectorized operations.
    Calculates centroid of image <img> using pixel values as weights.
    This is calculated using a weighted mean.
    """
    if img.size == 0: #If image is empty, just return 0, 0
        warnings.warn("ERROR: The image is empty!")
        return 0, 0
    if np.max(img) == np.min(img):
        #The code will fail if all pixels in the image have the same value,
        #because of division by 0 when dividing by img_normalized_sum.
        #Hence, just return the centre of the image
        warnings.warn("WARNING: The image has all constant values!")
        return img.shape[1]/2, img.shape[0]/2
    
    #First, apply min-max normalization to image <img>
    img_normalized = (img - np.min(img))/(np.max(img)-np.min(img))
    img_normalized_sum = np.sum(img_normalized) #sum of pixels in img_normalized
    img_normalized_transposed = img_normalized.T #transposed version of img_normalized
    
    y_grid = np.arange(0,img.shape[0]).astype(float) #array from 0 to img.shape[0]
    x_grid = np.arange(0,img.shape[1]).astype(float) #array from 0 to img.shape[1]
    
    x_cen = np.sum(img_normalized * x_grid) / img_normalized_sum
    y_cen = np.sum(img_normalized_transposed * y_grid) / img_normalized_sum
    
    if calculate_std == True:
        #effecive number of measurements
        n_eff = img_normalized_sum**2/np.sum(np.power(img_normalized, 2))
        
        y_grid_sq = np.power(np.arange(0,img.shape[0]) - y_cen, 2)
        x_grid_sq = np.power(np.arange(0,img.shape[1]) - x_cen, 2)
    
        x_numerator = np.sum(img_normalized * x_grid_sq)
        y_numerator = np.sum(img_normalized_transposed * y_grid_sq)
                
        x_std = np.sqrt(x_numerator/img_normalized_sum * n_eff/(n_eff-1))
        y_std = np.sqrt(y_numerator/img_normalized_sum * n_eff/(n_eff-1))
        return x_cen, y_cen, x_std, y_std
    else:
        return x_cen, y_cen

if __name__ == "__main__":
    filename = 'Images/20220413_141608.tif'
    #filename = 'Images/20220413_141650.tif'
    #filename = 'Images/20220413_143326.tif'
    img = read_12bit_image_saved_as_16bit(filename)
    img8 = (img / 2**4).astype(np.uint8)
    
    plt.figure()
    plt.imshow(img)
    plt.colorbar()
    plt.title('Original image')
    plt.show()
    
    #Apply Sobel operator, and get derivative, which is grad
    grad = apply_sobel_op(img)
    plt.figure()
    plt.imshow(grad)
    plt.colorbar()
    plt.title('Sobel operator')
    plt.show()
    
    #Make 8 bit version of grad
    grad_norm = grad/np.max(grad)
    grad8 = (grad_norm * 2**8).astype(np.uint8)
    print(np.max(grad8))
    
    #Find contours in grad
    contours, hierarchy = cv.findContours(grad8, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_SIMPLE)
    grad_copy = grad8.copy()
    print(len(contours))
    ###########################################################################
    #https://stackoverflow.com/a/36315105
    cv.drawContours(grad_copy, contours, -1, (255, 255, 255), 3)
    #The 2 lines below give the exact same results as the line above
    #for contour in contours:
    #    cv.drawContours(grad_copy, [contour], 0, (255, 255, 255), 3)
    ###########################################################################
    plt.imshow(grad_copy)
    plt.colorbar()
    plt.title('All contours in Sobel operator applied image')
    plt.show()
    
    #Find the largest contour among the contorus in grad
    c = max(contours, key = cv.contourArea)
    grad_copy2 = grad8.copy()
    cv.drawContours(grad_copy2, [c], -1, (255, 255, 255), 1)
    plt.figure()
    plt.imshow(grad_copy2)
    plt.colorbar()
    plt.title('Largest contour in Sobel operator applied image')
    plt.show()
    
    #thresh = otsu_method(img.astype(np.uint16), bit_depth=12)
    #print(thresh)
    
    #Apply Canny Edge detection to 8 bit version of img
    edges = cv.Canny(img8,100,200)
    plt.figure()
    plt.imshow(edges)
    plt.colorbar()
    plt.title('Canny Edge applied to image')
    plt.show()
    
    #Find the contours in Canny edge result
    contours_edge, _ = cv.findContours(edges, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_SIMPLE)
    img8_copy = img8.copy()
    cv.drawContours(img8_copy, contours_edge, -1, (255, 255, 255), 1)
    plt.imshow(img8_copy)
    plt.colorbar()
    plt.title('All contours from Canny edge')
    plt.show()
    
    #Find the largest contour among contours in Canny edge result
    contour_edge_max = max(contours_edge, key = cv.contourArea)
    img8_copy2 = img8.copy()
    cv.drawContours(img8_copy2, [contour_edge_max], 0, (255, 255, 255), 2)
    plt.figure()
    plt.imshow(img8_copy2)
    plt.colorbar()
    plt.title('Largest contour from Canny edge')
    plt.show()
    
    epsilon = 0.01*cv.arcLength(contour_edge_max,True)
    approx = cv.approxPolyDP(contour_edge_max,epsilon,True)
    img8_copy3 = img8.copy()
    cv.drawContours(img8_copy3, [approx], 0, (255, 255, 255), 1)
    plt.figure()
    plt.imshow(img8_copy3)
    plt.colorbar()
    plt.title('Largest contour from Canny edge, poly approx')
    plt.show()
    
    hull = cv.convexHull(contour_edge_max)
    img8_copy4 = img8.copy()
    cv.drawContours(img8_copy4, hull, -1, (255, 255, 255), 10)
    plt.figure()
    plt.imshow(img8_copy4)
    plt.colorbar()
    plt.title('Largest contour from Canny edge, convex hull')
    plt.show()
    
    x_cen, y_cen, x_cen_std, y_cen_std = get_image_centroid_vectorized(img, calculate_std=True)
    #print(x_cen, y_cen)
    plt.imshow(img)
    plt.colorbar()
    plt.errorbar(x_cen, y_cen, xerr=x_cen_std, yerr=y_cen_std, fmt='x', markersize=8, elinewidth=1, capsize=2, color='red', label='Centroid')
    plt.legend(framealpha=1.0)
    plt.title('Image centroid position')
    plt.show()
    
    
    ###################################################################################
    
    import sys
    sys.exit()
    
    test_filename = 'Images/shapes.png'
    test_img_PIL = Image.open(test_filename)
    test_img = np.asarray(test_img_PIL).astype(np.uint8)
    test_img = cv.cvtColor(test_img, cv.COLOR_RGB2GRAY)
    test_img = 255 - test_img
    plt.imshow(test_img)
    plt.show()
    contours, hierarchy = cv.findContours(test_img, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_SIMPLE)
    test_img_copy = test_img.copy()
    for contour in contours:
        cv.drawContours(test_img_copy, [contour], -1, (255, 0, 0), thickness=3)
    plt.imshow(test_img_copy)
    plt.show()
    

    #img8bit = (img / 2**4).astype(np.uint8)
    #print(img8bit)