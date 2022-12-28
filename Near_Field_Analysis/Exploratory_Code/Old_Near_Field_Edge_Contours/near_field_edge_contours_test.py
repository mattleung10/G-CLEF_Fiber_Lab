#Matthew Leung
#March 2022

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2 as cv
import scipy.ndimage
import warnings

#######################################################################################

def binary_circle(x, y, x0, y0, r):
    """Filled binary circle"""
    return np.where((x-x0)**2 + (y-y0)**2 < r**2, 1, 0)

#######################################################################################

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

#######################################################################################

def get_image_centroid_vectorized_no_norm(img):
    """
    Calculates centroid of image <img> using pixel values as weights.
    This is calculated using a weighted mean.
    No image normalization is applied.
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
    
    img_normalized = img #I don't want to rename the variables, so I just did this assignment
    img_normalized_sum = np.sum(img_normalized) #sum of pixels in img_normalized        
    img_normalized_transposed = img_normalized.T #transposed version of img_normalized
    
    y_grid = np.arange(0,img.shape[0]).astype(float) #array from 0 to img.shape[0]
    x_grid = np.arange(0,img.shape[1]).astype(float) #array from 0 to img.shape[1]
    
    x_cen = np.sum(img_normalized * x_grid) / img_normalized_sum    
    y_cen = np.sum(img_normalized_transposed * y_grid) / img_normalized_sum
    return x_cen, y_cen


def find_circle_from_binary_image(binary_img):
    """
    Calculates the centre and radius of a binary filled image of a circle.
    Assume that all nonzero pixels in binary_img belong to the circle, and all
    zero pixels in binary_img do not belong to the circle.
    """
    norm_binary_img = binary_img / np.max(binary_img) #normalized version of binary_img
    
    #Find the centroid of binary_img, weighted by pixel value
    x_cen, y_cen = get_image_centroid_vectorized_no_norm(norm_binary_img)
    
    area = np.sum(norm_binary_img) #area of circle
    radius = np.sqrt(area/np.pi)
    return x_cen, y_cen, radius

#######################################################################################

def process_contours_ellipse_output(ellipse):
    center, axes, angle = ellipse
    x0 = center[0]
    y0 = center[1]
    a = axes[0]
    b = axes[1]
    if a <= 0 or b <=0:
        return None
    area = np.pi * axes[0] * axes[1]
    eccentricity2 = 1 - b**2/a**2
    return x0, y0, a, b, angle, area, eccentricity2

def find_edge_contours(img):
    #https://docs.opencv.org/3.4/d4/d73/tutorial_py_contours_begin.html
    #https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contour_features/py_contour_features.html
    #https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
    #https://stackoverflow.com/questions/38530899/python-fit-ellipse-to-an-image
    
    #Gaussian blur the image to smooth out noise
    blurred_img = cv.GaussianBlur(img, (5, 5), 0)

    plt.figure()
    plt.hist(blurred_img.ravel(), 256)
    plt.show()
    
    _, thresh = cv.threshold(blurred_img, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    plt.figure()
    plt.imshow(thresh)
    plt.show()
    
    thresh2 = cv.adaptiveThreshold(blurred_img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 45, 1)
    plt.figure()
    plt.imshow(thresh2)
    plt.title('adaptive threshold')
    plt.show()
    
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    ellipses = []
    for i in range(0,len(contours),1):
        cnt = contours[i]
        #cv.drawContours(img, [cnt], 0, (255,255,255),3)
        if cnt.shape[0] >= 5: #Must have at least 5 points to fit ellipse
            ellipses += [cv.fitEllipse(cnt)]
            #print(ellipses[-1])
            #cv.ellipse(img, ellipses[-1], (255,255,255),2)
    #cv.imshow('dc', img)
    #cv.waitKey(0)
    
    plt.figure()
    plt.imshow(img)
    for ellipse in ellipses:
        r = process_contours_ellipse_output(ellipse)
        if r is None:
            continue
        x0, y0, a, b, angle, area, eccentricity2 = r
        ell = patches.Ellipse((x0, y0), a, b, angle, linewidth=1, edgecolor='r', facecolor='none')
        plt.gca().add_patch(ell)
    plt.show()
    
def find_edge(img):
    #Gaussian blur the image to smooth out noise
    blurred_img = cv.GaussianBlur(img, (31, 31), 0)
    
    #Apply Sobel operator
    grad = apply_sobel_op(blurred_img)
    
    mask = np.where(grad > 0)
    nonzero_grad_vals = grad[mask] #np.ndarray containing the nonzero values in grad
    mean_grad = np.mean(nonzero_grad_vals) #average of the nonzero values in grad
    
    #Create a binary image <thresholded_grad>, where all pixels above <threshold_val>
    #in <grad> are forced to 255, and below <threshold_val> are forced to 0
    threshold_val = mean_grad
    threshold_val = 0
    thresholded_grad = np.where(grad > threshold_val, 255, 0)
    
    #Fill holes in binary image <thresholded_grad>
    filled_binary_img = scipy.ndimage.morphology.binary_fill_holes(thresholded_grad)
    
    #Find the centre and radius of filled circle
    x_cen, y_cen, radius = find_circle_from_binary_image(filled_binary_img)
    return x_cen, y_cen, radius

#######################################################################################
    
def plot_edge(img, x_cen, y_cen, radius):
    h, w = img.shape
    x = np.arange(0, w)
    y = np.arange(0, h)
    X, Y = np.meshgrid(x, y)
    
    plt.figure()
    plt.imshow(img)
    plt.colorbar()
    circle1 = plt.Circle((x_cen, y_cen), radius=radius, color='r', fill=False)
    plt.gca().add_patch(circle1)
    plt.show()
    
if __name__ == "__main__":
    #filename = 'Images/220301_1634 NF of 200um Fiber, Laser Source.jpg'
    filename = 'Images/220301_1635 NF of 200um Fiber, Laser Source.jpg'
    #filename = 'Images/220301_1626 NF of 200um Fiber, Laser Source.jpg'
    filename = 'Images/20220310 1128 Far Field.png'
    filename = 'Images/20220310 1130 Far Field No ND.png'
    img = cv.imread(filename, cv.IMREAD_GRAYSCALE)
    noise = np.round(np.random.normal(25,5,img.shape).reshape(img.shape))
    noise = noise.astype('uint8')
    #img += noise
    
    ###########################################################################
    #Gaussian blur the image to smooth out noise
    blurred_img = cv.GaussianBlur(img, (5, 5), 0)
    #Apply Sobel operator
    grad = apply_sobel_op(blurred_img)
    plt.figure()
    plt.imshow(grad)
    plt.show()
    blurred_img = grad
    plt.figure()
    plt.hist(blurred_img.ravel(), int(np.max(blurred_img))+1)
    plt.show()
    grad_32bit = (grad / np.max(grad) * 2**8).astype(np.uint8)
    
    thresh2 = cv.adaptiveThreshold(grad_32bit, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 45, 1)
    plt.figure()
    plt.imshow(thresh2)
    plt.title('adaptive threshold')
    plt.show()
    
    sys.exit()
    ###########################################################################
    
    
    find_edge_contours(img)
    
    x_cen, y_cen, radius = find_edge(img)
    print(radius)
    plot_edge(img, x_cen, y_cen, radius)
    
