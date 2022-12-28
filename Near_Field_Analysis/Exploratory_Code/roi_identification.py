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

def driver(img, verbosity=0):
    """
    img is in 12 bit
    """
    if verbosity >= 2:
        #Plot original image
        plt.imshow(img)
        plt.colorbar()
        plt.title('Original Image')
        plt.show()
    
    #Create 8 bit version of original image, for use with Canny Edge
    img8 = (img / 2**4).astype(np.uint8)

    #Apply Canny Edge Detection to 8 bit version of img
    minVal = np.max(img8) / 2
    maxVal = np.max(img8) * 3/4
    edges = cv.Canny(img8, minVal, maxVal)
    if verbosity >= 2:
        plt.imshow(edges)
        plt.colorbar()
        plt.title('Canny Edge applied to Original Image')
        plt.show()
    
    #Find the contours in the result of Canny Edge
    contours_ce, hierarchy = cv.findContours(edges, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_SIMPLE)
    if verbosity >= 2:
        img8_copy = img8.copy()
        cv.drawContours(img8_copy, contours_ce, -1, (255, 255, 255), 1)
        plt.imshow(img8_copy)
        plt.colorbar()

        plt.title('All Contours from Canny Edge')
        plt.show()
    
    #Find the contour with the largest area
    contour_ce_largest = max(contours_ce, key = cv.contourArea)
    if verbosity >= 2:
        img8_copy2 = img8.copy()
        contour_ce_largest_unzipped = list(zip(*contour_ce_largest[:,0,:]))
        #cv.drawContours(img8_copy2, [contour_ce_largest], 0, (255, 255, 255), 1)
        plt.figure()
        plt.imshow(img8_copy2)
        #Plot line from first point to last point in CW direction
        plt.plot(contour_ce_largest_unzipped[0], contour_ce_largest_unzipped[1], color='red')
        #Plot line from first point to last point, complete the shape
        plt.plot([contour_ce_largest_unzipped[0][0],contour_ce_largest_unzipped[0][-1]], [contour_ce_largest_unzipped[1][1],contour_ce_largest_unzipped[1][-1]], color='red')
        plt.colorbar()
        plt.title('Largest Contour from Canny Edge')
        plt.show()
    
    #Find the Convex Hull of the largest contour
    hull = cv.convexHull(contour_ce_largest)
    list_of_points = hull[:,0,:] #list of points in Convex Hull
    unzipped = list(zip(*list_of_points))
    if verbosity >= 1:    
        plt.figure()
        plt.imshow(img)
        plt.colorbar()
        plt.plot(unzipped[0], unzipped[1], marker='o', markersize=5, color='red')
        plt.plot([unzipped[0][0],unzipped[0][-1]], [unzipped[1][1],unzipped[1][-1]], color='red')
        plt.title('Convex Hull of Largest Contour')
        plt.show()
    
    #Create a binary mask for points inside the Convex Hull
    hull_mask = np.zeros(shape=np.shape(img8)) #Start with an empty array
    cv.drawContours(hull_mask, [contour_ce_largest], 0, (255, 255, 255), -1) #draw on the empty array the points inside the Convex Hull
    hull_mask /= 255 #Normalize the array to 1
    plt.imshow(hull_mask)
    plt.colorbar()
    plt.plot(unzipped[0], unzipped[1], marker='o', markersize=5, color='red')
    plt.plot([unzipped[0][0],unzipped[0][-1]], [unzipped[1][1],unzipped[1][-1]], color='red')
    plt.title('Binary Mask of Points inside Convex Hull')
    plt.show()
    
    #Find the points in the original image which are inside the Convex Hull
    img_points_in_hull = np.where(hull_mask==1, img, 0) #This array is the original image, except all points outside the Convex Hull are forced to 0
    plt.imshow(img_points_in_hull)
    plt.colorbar()
    plt.title('Points of Original Image inside Convex Hull')
    plt.show()
    
    #Find the points in the original image which were not in the Convex Hull
    residual = img - img_points_in_hull
    plt.imshow(residual)
    plt.colorbar()
    plt.title('Residuals')
    plt.show()
    
    #Obtain the pixel values which are inside the Convex Hull
    whered = np.where(hull_mask==1, img, -1000) #-1000 can instead be any arbitrary number which is NOT a valid pixel value
    list_of_pixel_vals_in_hull = whered[np.where(whered != -1000)]
    print("There are {} pixels inside the Convex Hull".format(len(list_of_pixel_vals_in_hull)))
    mu = np.mean(list_of_pixel_vals_in_hull)
    sigma = np.std(list_of_pixel_vals_in_hull)
    print("Image metrics of these pixels:")
    print("mu = {}".format(mu))
    print("sigma = {}".format(sigma))
    print("mu/sigma = {}".format(mu/sigma))
    

if __name__ == "__main__":
    filename = 'Images/20220413_141608.tif'
    filename = 'Images/20220323_1155 Near Field Rect Laser.tif'
    #filename = 'Images/20220413_141650.tif'
    #filename = 'Images/20220413_143326.tif'
    #filename = 'Images/20220418_153630.tif' #rectangular fiber, mode scrambler off
    #filename = 'Images/20220418_153752.tif' #rectangular fiber, mode scrambler on
    filename = 'Images/20220418_163807.tif' #square fiber, mode scrambler off
    img_og = read_12bit_image_saved_as_16bit(filename)
    xmin = 225
    xmax = 400
    ymin = 800
    ymax = 975
    img = img_og[xmin:xmax, ymin:ymax]
    img = img_og
    driver(img, verbosity=2)
