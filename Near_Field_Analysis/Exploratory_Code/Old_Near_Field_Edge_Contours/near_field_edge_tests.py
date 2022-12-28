#Matthew Leung
#March 2022

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import warnings

def take_img_slice(img):
    h, w = img.shape

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

def binary_circle(x, y, x0, y0, r):
    return np.where((x-x0)**2 + (y-y0)**2 < r**2, 1, 0)

def _binary_circle(M, x0, y0, r):
    x, y = M
    return binary_circle(x, y, x0, y0, r)

def fit_binary_circle(binary_img, p0, bounds):
    
    h, w = binary_img.shape
    norm_binary_img = binary_img / np.max(binary_img)
    
    fit_fcn = _binary_circle
    
    #2D domain of the fit
    x = np.arange(0, w)
    y = np.arange(0, h)
    X, Y = np.meshgrid(x, y)
    Z = norm_binary_img

    #Ravel the meshgrids of XY points to a pair of 1D arrays
    xdata = np.vstack((X.ravel(), Y.ravel()))
    
    #Use flattened (ravelled) ordering of the data points
    params, cov = scipy.optimize.curve_fit(fit_fcn, xdata, Z.ravel(), p0=p0, bounds=bounds, maxfev=2e4)
    return params, cov

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
    norm_binary_img = binary_img / np.max(binary_img)
    x_cen, y_cen = get_image_centroid_vectorized_no_norm(norm_binary_img)
    
    area = np.sum(norm_binary_img)
    radius = np.sqrt(area/np.pi)
    return x_cen, y_cen, radius


if __name__ == "__main__":
    #filename = 'Images/220301_1634 NF of 200um Fiber, Laser Source.jpg'
    filename = 'Images/220301_1635 NF of 200um Fiber, Laser Source.jpg'
    #filename = 'Images/220301_1626 NF of 200um Fiber, Laser Source.jpg'
    
    np.random.seed(100)
    img = cv.imread(filename, cv.IMREAD_GRAYSCALE)
    noise = np.round(np.random.normal(25,5,img.shape).reshape(img.shape))
    noise = noise.astype('uint8')
    print(noise)
    img += noise
    blur = cv.GaussianBlur(img,(31,31),0)
    filtered = img - blur
    filtered += 127

    plt.figure()
    plt.imshow(img)
    plt.colorbar()
    plt.show()
    
    plt.figure()
    plt.imshow(filtered)
    plt.colorbar()
    plt.show()
    
    blurred_img = cv.GaussianBlur(img, (31, 31), 0)
    plt.imshow(blurred_img)
    plt.show()
    
    grad = apply_sobel_op(blurred_img)
    
    mask = np.where(grad > 0)
    nonzero_grad_vals = grad[mask]
    print(nonzero_grad_vals, nonzero_grad_vals.shape)
    print(grad.shape, grad.size)
    mean_grad = np.mean(nonzero_grad_vals)
    std_grad = np.std(nonzero_grad_vals)
    print(mean_grad)
    print(std_grad)
    
    
    plt.imshow(grad)
    plt.colorbar()
    plt.title('Sobel')
    plt.show()
    
    filtered_grad = np.where(grad > mean_grad, 255, 0)
    plt.imshow(filtered_grad)
    plt.colorbar()
    plt.title('Thresholded Sobel')
    plt.show()
    
    import scipy.ndimage
    a = scipy.ndimage.morphology.binary_fill_holes(filtered_grad)
    plt.imshow(a)
    plt.title('Filled thresholded Sobel')
    plt.show()
    
    '''
    #Plot binary circle
    h, w = img.shape
    x = np.arange(0, w)
    y = np.arange(0, h)
    X, Y = np.meshgrid(x, y)
    Z = binary_circle(X, Y, 600, 400, 200)
    plt.imshow(Z)
    plt.show()
    '''
    
    """
    h, w = img.shape
    p0 = [w/2, h/2, 200]
    bounds = ((0,0,0),(w,h,np.max([w,h])))
    params, params_cov = fit_binary_circle(a, p0, bounds)
    x = np.arange(0, w)
    y = np.arange(0, h)
    X, Y = np.meshgrid(x, y)
    Z = binary_circle(X, Y, *params)
    plt.imshow(Z)
    plt.title('Fit')
    plt.show()
    """
    print("Radius is {} pixels".format(params[2]))
    
    h, w = img.shape
    x = np.arange(0, w)
    y = np.arange(0, h)
    X, Y = np.meshgrid(x, y)
    params = find_circle_from_binary_image(a)
    Z = binary_circle(X, Y, *params)
    plt.imshow(Z)
    plt.title('Fit')
    plt.show()
    
    plt.imshow(a-Z)
    plt.title('Residuals')
    plt.colorbar()
    plt.show()
    
def other2():
    #OpenCV contours
    
    thresh = cv.convertScaleAbs(filtered_grad)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    canvas = np.zeros(img.shape)
    cv.drawContours(canvas, contours, -1, (255,255,255), -1)
    plt.imshow(canvas)
    plt.show()
    
    
    
def lapla():
    #Lapacian operator
    
    ddepth = cv.CV_64F
    lap =  cv.Laplacian(img, ddepth, ksize=3)
    plt.imshow(np.abs(lap))
    plt.title('Laplacian')
    plt.colorbar()
    plt.show()
    
def other():
    #Hough circle transform
    
    edges = grad
    from skimage.transform import hough_circle, hough_circle_peaks
    from skimage.draw import circle_perimeter
    from skimage import data, color
    # Detect two radii
    hough_radii = np.arange(200, 220, 4)
    hough_res = hough_circle(edges, hough_radii)
    
    # Select the most prominent 3 circles
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
                                               total_num_peaks=3)
    
    # Draw them
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 4))
    img = color.gray2rgb(img)
    for center_y, center_x, radius in zip(cy, cx, radii):
        circy, circx = circle_perimeter(center_y, center_x, radius,
                                        shape=img.shape)
        img[circy, circx] = (220, 20, 20)
    
    ax.imshow(img, cmap=plt.cm.gray)
    plt.show()

    import sys
    sys.exit()

    cimg = cv.cvtColor(img,cv.COLOR_GRAY2BGR)
    edges = cv.Canny(img,35,100)
    '''
    plt.imshow(edges)
    plt.show()
    '''
    circles = cv.HoughCircles(grad,cv.HOUGH_GRADIENT,1,minDist=100,
                                minRadius=100,maxRadius=500)
    circles = np.uint16(np.around(circles))
    
    for i in circles[0,:]:
        # draw the outer circle
        cv.circle(cimg,(i[0],i[1]),i[2],(255,0,0),2)
        # draw the center of the circle
        cv.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
    
    plt.imshow(cimg)
    plt.show()
