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


if __name__ == "__main__":
    filename = 'Images/20220413_141608.tif'
    filename = 'Images/20220323_1155 Near Field Rect Laser.tif'
    #filename = 'Images/20220413_141650.tif'
    #filename = 'Images/20220413_143326.tif'
    filename = 'Images/20220418_153630.tif' #rectangular fiber, mode scrambler off
    #filename = 'Images/20220418_153752.tif' #rectangular fiber, mode scrambler on
    #filename = 'Images/20220418_163807.tif' #square fiber, mode scrambler off
    img = read_12bit_image_saved_as_16bit(filename)
    
    img8 = (img / 2**4).astype(np.uint8)
    
    plt.figure()
    plt.imshow(img)
    plt.colorbar()
    plt.title('Original image')
    plt.show()
    
    smoothed = cv.GaussianBlur(img8, (3, 3), 0)
    #smoothed = img8
    #Apply Sobel operator, and get derivative, which is grad
    grad = apply_sobel_op(smoothed)
    #grad = np.abs(cv.Laplacian(img, ddepth=cv.CV_64F, ksize=3))
    plt.figure()
    plt.imshow(grad)
    plt.colorbar()
    plt.title('Sobel operator')
    plt.show()
    
    #Make 8 bit version of grad
    grad_norm = grad/np.max(grad)
    grad8 = (grad_norm * (2**8-1)).astype(np.uint8)
    print(np.max(grad8))
    plt.imshow(grad8)
    plt.colorbar()
    plt.title('Sobel operator NORMALIZED')
    plt.show()
    
    grad_thresh = np.where(grad_norm > 0, 255, 0)
    plt.imshow(grad_thresh)
    plt.colorbar()
    plt.title('Gradient THRESHOLDED')
    plt.show()
    
    #Apply Canny Edge Detection to 8 bit version of grad
    minVal = np.max(grad8) / 4
    maxVal = np.max(grad8) * 3/4
    edges = cv.Canny(grad8, minVal, maxVal)
    
    plt.imshow(edges)
    plt.colorbar()
    plt.title('Canny Edge applied to Grad')
    plt.show()
    
    contours_ce, hierarchy = cv.findContours(edges, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_NONE)
    grad8_copy = grad8.copy()
    cv.drawContours(grad8_copy, contours_ce, -1, (255, 255, 255), 1)
    plt.imshow(grad8_copy)
    plt.colorbar()
    plt.title('All Contours from Canny Edge')
    plt.show()
    
    #Find the contour with the largest area
    contour_ce_largest = max(contours_ce, key = cv.contourArea)
    img8_copy2 = img8.copy()
    contour_ce_largest_unzipped = list(zip(*contour_ce_largest[:,0,:]))
    #cv.drawContours(img8_copy2, [contour_ce_largest], 0, (255, 255, 255), 1)
    plt.figure()
    plt.imshow(img8_copy2)
    #Plot line from first point to last point in CW direction
    x = np.array(contour_ce_largest_unzipped[0])
    y = np.array(contour_ce_largest_unzipped[1])
    plt.plot(contour_ce_largest_unzipped[0], contour_ce_largest_unzipped[1], color='red')
    #Plot line from first point to last point, complete the shape
    plt.plot([contour_ce_largest_unzipped[0][0],contour_ce_largest_unzipped[0][-1]], [contour_ce_largest_unzipped[1][1],contour_ce_largest_unzipped[1][-1]], color='red')
    plt.colorbar()
    plt.quiver(x[:-1], y[:-1], x[1:]-x[:-1], y[1:]-y[:-1], scale_units='xy', angles='xy', color='orange', scale=1)
    plt.title('Largest Contour from Canny Edge')
    plt.show()
    
    blank = np.zeros(shape=img8.shape)
    cv.drawContours(blank, [contour_ce_largest], -1, 255, 1)
    plt.imshow(blank)
    plt.show()
    
    #epsilon = 0.001*cv.arcLength(contour_ce_largest ,True)
    #approx = cv.approxPolyDP(contour_ce_largest,epsilon,True)
    #blank2 = np.zeros(shape=img8.shape)
    #cv.drawContours(blank2, [approx], -1, (255, 255, 255), 1)
    #plt.imshow(blank2)
    #plt.colorbar()
    #plt.title('DP')
    #plt.show()
    
    #import skimage
    #filled = skimage.segmentation.flood_fill(blank, (600,1000), 127)
    #plt.imshow(filled, vmin=0)
    #plt.colorbar()
    #plt.show()
    
    import sys
    sys.exit()
    ######################################################################################################################################################
    ######################################################################################################################################################
    
    
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
    
    img8_copy = img8.copy()
    cv.drawContours(img8_copy, [c], -1, (255, 255, 255), 1)
    plt.figure()
    plt.imshow(img8_copy)
    plt.colorbar()
    plt.show()
    
    smoothed = cv.GaussianBlur(img8, (3, 3), 0)
    #Apply Canny Edge detection to 8 bit version of img
    edges = cv.Canny(smoothed,100,200)
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