#Matthew Leung
#Code last updated: February 28, 2022
"""
Same as the Gaussian_Fit.py script in the Gaussian_Fit directory, but with some
modifications. E.g. the option to fit a clipped Gaussian was added.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scipy.optimize
import scipy.special
import cv2 as cv
import warnings

#######################################################################################
#######################################################################################
#GAUSSIAN FUNCTIONS

def gaussian(x, y, x0, y0, sigma_x, sigma_y, A, back, theta):
    #2D Gaussian
    xs = x - x0
    ys = y - y0
    xr = xs*np.cos(theta) + ys*np.sin(theta)
    yr = -xs*np.sin(theta) + ys*np.cos(theta)
    return A * np.exp( -xr**2/(2*sigma_x**2) -yr**2/(2*sigma_y**2)) + back

def _gaussian(M, x0, y0, sigma_x, sigma_y, A, back, theta):
    #This is just a wrapper function
    x, y = M
    return gaussian(x, y, x0, y0, sigma_x, sigma_y, A, back, theta)

def gaussian_clipped(x, y, x0, y0, sigma_x, sigma_y, A, back, theta):
    #Clip all pixels above clip_lim
    clip_lim = 255
    g = gaussian(x, y, x0, y0, sigma_x, sigma_y, A, back, theta)
    return np.where(g > clip_lim, clip_lim, g)

def _gaussian_clipped(M, x0, y0, sigma_x, sigma_y, A, back, theta):
    #This is just a wrapper function
    x, y = M
    return gaussian_clipped(x, y, x0, y0, sigma_x, sigma_y, A, back, theta)

def gaussian_FWHM(sigma):
    return 2*sigma*np.sqrt(2*np.log(2))

def gaussian_intensity(A, sigma_x, sigma_y, A_err, sigma_x_err, sigma_y_err):
    I = 2*np.pi * A * sigma_x * sigma_y
    I_err = I * np.sqrt((A_err/A)**2 + (sigma_x_err/sigma_x)**2 + (sigma_y_err/sigma_y)**2)
    return I, I_err

def gaussian_FW_at_val(A, sigma, val):
    #Computes the full width of the Gaussian at a height of <val>
    if val >= A:
        return 0
    else:
        return 2*sigma*np.sqrt(-2*np.log(val/A))

#######################################################################################
#######################################################################################
#FUNCTIONS TO PROCESS BOUNDS, PARAMETERS, AND RESULTS

def get_param_names(fit_fcn):
    """
    Get the names of the parameters of fit_fcn
    Returns a list of str
    """
    from inspect import signature
    sig = signature(fit_fcn)
    sig_str = str(sig)
    param_names = ''.join(c for c in sig_str if c not in '() ')
    param_names = param_names.split(',')
    return param_names

def construct_bounds_p0(fit_fcn, lb_dict, ub_dict, p0_dict):
    """Constructs bounds and p0"""
    param_names = get_param_names(fit_fcn)
    lb = []
    ub = []
    p0 = []
    for i in range(1,len(param_names),1):
        param_name = param_names[i] #this is a str
        lb += [lb_dict[param_name]]
        ub += [ub_dict[param_name]]
        p0 += [p0_dict[param_name]]
    bounds = (tuple(lb),tuple(ub))
    return bounds, p0

def construct_params_result_dicts(fit_fcn, params, params_uncert):
    """Constructs dicts representing the fitted parameters and uncerts"""
    param_names = get_param_names(fit_fcn)
    params_dict = dict()
    params_uncert_dict = dict()
    for i in range(1,len(param_names),1):
        param_name = param_names[i]
        params_dict[param_name] = params[i-1]
        params_uncert_dict[param_name] = params_uncert[i-1]
    return params_dict, params_uncert_dict

def construct_params_from_params_dict(fit_fcn, params_dict):
    """Constructs list representing fitted parameters, from a params_dict"""
    param_names = get_param_names(fit_fcn)
    params = []
    for i in range(1,len(param_names),1):
        param_name = param_names[i]
        params += [params_dict[param_name]] 
    return params

def display_results(params_dict, params_uncert_dict):
    """Gets list representation of fit params"""
    results_text = []
    for k in params_dict.keys():
        results_text += [k + ' = {} ± {}'.format(params_dict[k], params_uncert_dict[k])]
    return results_text

#######################################################################################
#######################################################################################

def fit_function_to_img(img, fit_fcn, p0, bounds):
    """
    Fits a function <fit_fcn> to an image <img>, given initial parameters <p0>
    and bounds <bounds>
    """
    
    #2D domain of the fit
    x = np.arange(0, img.shape[1])
    y = np.arange(0, img.shape[0])
    X, Y = np.meshgrid(x, y)

    #Z = abs(img)
    Z = img
    Z_uncert = np.sqrt(np.abs(img)) #uncertainty is square root of counts
    
    #Do this to avoid ValueError: Residuals are not finite in the initial point
    Z_uncert[Z_uncert == 0] = 1 #NEW! February 28, 2022 <<<<<<<<<<<<<<<<<<<<<<<<<<<=====================================

    ########################################################
    #Simple Fit Code:

    #Ravel the meshgrids of XY points to a pair of 1D arrays
    xdata = np.vstack((X.ravel(), Y.ravel()))
    
    ###########################################################################
    #Find the number of fit parameters in function and also the dof
    #See: https://stackoverflow.com/questions/847936/how-can-i-find-the-number-of-arguments-of-a-python-function/41188411#41188411
    from inspect import signature
    sig = signature(fit_fcn)
    dof = xdata.size - (len(sig.parameters) - 1)
    ###########################################################################
    
    #Use flattened (ravelled) ordering of the data points
    params, cov = scipy.optimize.curve_fit(fit_fcn, xdata, Z.ravel(), p0=p0, sigma=Z_uncert.ravel(), bounds=bounds, maxfev=2e4)

    # Calculate Chi-squared
    chisq = sum(((Z.ravel()-fit_fcn(xdata,*params))/Z_uncert.ravel())**2)
    # Convert Scipy cov matrix to standard covariance matrix.
    cov = cov*dof/chisq
    p = params  
        
    p_uncert = [] #store the uncertainty in the parameters
    for i in range(len(p)) :
        p_uncert += [cov[i,i]**0.5*max(1,np.sqrt(chisq/dof))]
    cdf = scipy.special.chdtrc(dof,chisq)
    chisq_per_dof = chisq/dof
    #print("Chi^2/dof = {}".format(chisq_per_dof))
    #print("CDF = {}".format(cdf))
    
    return params, p_uncert

def eval_fit_fcn_for_img(data, fit_fcn, params):
    x = np.arange(0, data.shape[1])
    y = np.arange(0, data.shape[0])
    X, Y = np.meshgrid(x, y)
    Z_fit = fit_fcn([X, Y], *params)
    return Z_fit

def fit_gaussian(data, xy_guess=None, clipped_gaussian=False):
    """
    Fit Gaussian PSF to image <img>
    """
    img = data
    theta_min = 0
    theta_max = np.pi
    theta_init = 0
    lb_dict = {'x0':0, 'y0':0, 'sigma_x':1, 'sigma_y':1, 'A':0, 'back':0, 'theta':theta_min}
    ub_dict = {'x0':img.shape[1], 'y0':img.shape[0], 'sigma_x':img.shape[0], 'sigma_y':img.shape[0], 'A':np.inf, 'back':np.inf, 'theta':theta_max}
    p0_dict = {'x0':img.shape[1]/2, 'y0':img.shape[0]/2, 'sigma_x':4, 'sigma_y':4, 'A':np.max(img)*2, 'back':1.1*np.min(img), 'theta':theta_init}
    
    if xy_guess is not None:
        p0_dict['x0'] = xy_guess[0]
        p0_dict['y0'] = xy_guess[1]
    
    #fit_fcn = lambda M, x0, y0, sigma_x, sigma_y, A, theta: _gaussian(M, x0, y0, sigma_x, sigma_y, A, 0, theta)
    if clipped_gaussian == False:
        fit_fcn = _gaussian
    else:
        fit_fcn = _gaussian_clipped
    bounds, p0 = construct_bounds_p0(fit_fcn, lb_dict, ub_dict, p0_dict)
    params, params_uncert = fit_function_to_img(img, fit_fcn, p0, bounds)
    params_dict, params_uncert_dict = construct_params_result_dicts(fit_fcn, params, params_uncert)
    return params, params_uncert, params_dict, params_uncert_dict, fit_fcn

#######################################################################################
#######################################################################################

def get_image_centroid(img, calculate_std=False):
    """
    Calculates centroid of image <img> using pixel values as weights.
    This is calculated using a weighted mean.
    This function assumes that img does not have any issues (e.g. it's not empty).
    """
    #First, apply min-max normalization to image <img>
    img_normalized = (img - np.min(img))/(np.max(img)-np.min(img))
    x_cen = 0
    y_cen = 0
    for i in range(0,img_normalized.shape[0],1):
        for j in range(0,img_normalized.shape[1],1):
            weight = img_normalized[i,j]
            x_cen += j*weight
            y_cen += i*weight
    x_cen /= np.sum(img_normalized)
    y_cen /= np.sum(img_normalized)
    
    if calculate_std == True:
        #effecive number of measurements
        n_eff = np.sum(img_normalized)**2/np.sum(np.power(img_normalized, 2))
        
        x_numerator = 0
        y_numerator = 0
        for i in range(0,img_normalized.shape[0],1):
            for j in range(0,img_normalized.shape[1],1):
                weight = img_normalized[i,j]
                x_numerator += weight*(j-x_cen)**2
                y_numerator += weight*(i-y_cen)**2
                
        x_std = np.sqrt(x_numerator/np.sum(img_normalized) * n_eff/(n_eff-1))
        y_std = np.sqrt(y_numerator/np.sum(img_normalized) * n_eff/(n_eff-1))
        return x_cen, y_cen, x_std, y_std
    else:
        return x_cen, y_cen

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

def get_image_centroid_vectorized_no_norm(img):
    """
    Same as get_image_centroid_vectorized function except that the weights
    are not normalized first. Normalization is only needed if we want to
    compute the standard deviations of the centroid.
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
    
    img_normalized = img #I don't want to rename the variables, so I just did this assignment
    img_normalized_sum = np.sum(img_normalized) #sum of pixels in img_normalized        
    img_normalized_transposed = img_normalized.T #transposed version of img_normalized
    
    y_grid = np.arange(0,img.shape[0]).astype(float) #array from 0 to img.shape[0]
    x_grid = np.arange(0,img.shape[1]).astype(float) #array from 0 to img.shape[1]
    
    x_cen = np.sum(img_normalized * x_grid) / img_normalized_sum    
    y_cen = np.sum(img_normalized_transposed * y_grid) / img_normalized_sum
    return x_cen, y_cen

def center_of_mass(img):
    """
    This function is slightly modified from the scipy function
    scipy.ndimage.measurements.center_of_mass
    Calculates the center of mass of the values of an array.
    """
    normalizer = np.sum(img)
    grids = np.ogrid[[slice(0, i) for i in img.shape]]

    results = [np.sum(img * grids[direction].astype(float)) / normalizer
               for direction in range(img.ndim)]
    
    return tuple(results)

#######################################################################################
#######################################################################################

def crop_image(data, x, y, zoom_size=200):
    """
    Crops an image (np.ndarray) so that it is centred at (x,y) and has a radius
    of <zoom_size>
    INPUTS:
        ::np.ndarray:: data
        ::int:: x
        ::int:: y
        ::int:: zoom_size #this is in pixels
    """
    x_min = int(x - zoom_size)
    x_max = int(x + zoom_size)
    y_min = int(y - zoom_size)
    y_max = int(y + zoom_size)
    zoomed_in_data = data[y_min:y_max, x_min:x_max]
    return zoomed_in_data

def crop_image_safe(data, x, y, zoom_size=200):
    """
    Crops an image (np.ndarray) so that it is centred at (x,y) and has a radius
    of <zoom_size>. Safe version of crop_imagewhich ensures that no empty
    slices will be taken.
    INPUTS:
        ::np.ndarray:: data
        ::int:: x
        ::int:: y
        ::int:: zoom_size #this is in pixels
    """
    #Do not let x_min, x_max, y_min, y_max go out of bounds
    x_min = np.max([int(x - zoom_size), 0])
    x_max = np.min([int(x + zoom_size), data.shape[1]])
    y_min = np.max([int(y - zoom_size), 0])
    y_max = np.min([int(y + zoom_size), data.shape[0]])
    
    #Make sure that x_min != x_max and y_min != y_max to prevent empty slices
    #(e.g. that the dimension of the cropped image won't be 0 in any directions)
    if x_min == x_max:
        if x_min == 0: #if centre is at left edge
            x_max += 1
        elif x_max == data.shape[1]-1: #if centre is at right edge
            x_min -= 1
        else: #just add this in case
            x_max += 1
    if y_min == y_max: 
        if y_min == 0: #if centre is at top edge
            y_max += 1
        elif y_max == data.shape[0]-1: #if centre is at bottom edge
            y_min -= 1
        else: #just add this in case
            y_min -= 1
    zoomed_in_data = data[y_min:y_max, x_min:x_max]
    return zoomed_in_data

#######################################################################################
#######################################################################################

def find_centroid_iter(img, cen_tol=5, maxiter=10, do_plot=False):
    """
    Find the centroid of an image, by iteratively cropping the image and
    finding the centroid of the cropped image. Employs image clipping to
    disregard small pixel values (background noise).
    This function returns when the centroid has not change from the previous
    iteration within a tolerance of <cen_tol> pixels, or if the maximum number
    of iterations <maxiter> has been reached.
    INPUTS:
        ::np.ndarray:: img
        ::float:: cen_tol
        ::int:: maxiter
        ::boolean:: do_plot
    """
    mean = np.mean(img) #mean of the image
    std = np.std(img) #standard deviation of the image
    
    #clip the image so that values below mean+3*std are forced to 0
    for i in range(3,0,-1):
        clipped_img = np.where(img < mean+i*std, 0, img)
        if np.max(clipped_img) != np.min(clipped_img):
            break
    #clipped_img = np.where(img < mean+3*std, 0, img)
    
    x_cen, y_cen = get_image_centroid_vectorized(clipped_img) #initial centroid value
    prev_cen_values = {'x':x_cen, 'y':y_cen} #dictionary to store previous centroid values
    
    smallest_dist_to_edge = np.min([x_cen, img.shape[1]-x_cen, y_cen, img.shape[0]-y_cen]) #smallest distance from centroid to edge of an image
    radius = np.min([np.min(img.shape)/2, smallest_dist_to_edge]) #initialize the radius value to crop image
    radius = np.max([radius, 5]) #to avoid errors when cropping image, set the minimum allowed radius to be 5
    ctr = 0 #counter for the number of iterations
    while True:
        cropped_img = crop_image_safe(img, x_cen, y_cen, zoom_size=radius) #crop the image        
        
        #Calculate image stats and use them to clip image
        mean = np.mean(cropped_img)
        std = np.std(cropped_img)
        #Iteratively decrease the clipping threshold.
        #This is necessary because if mean+i*std is too large, then clipped_img
        #will have all the same pixel values and the code will error out
        for i in range(3,0,-1):
            clipped_img = np.where(cropped_img < mean+i*std, 0, cropped_img)
            if np.max(clipped_img) != np.min(clipped_img):
                break
        
        #Find the centroid in the coordinate system of cropped_img
        x_cen_prime, y_cen_prime = get_image_centroid_vectorized(clipped_img)
        #Transform the coordinates back to that of original uncropped img
        new_x_cen = x_cen - radius + x_cen_prime
        new_y_cen = y_cen - radius + y_cen_prime
        
        if do_plot == True: #create plots
            plt.figure()
            plt.imshow(img)
            plt.colorbar()
            rect = patches.Rectangle((x_cen-radius, y_cen-radius), radius*2, radius*2, linewidth=1, edgecolor='r', facecolor='none')
            plt.gca().add_patch(rect)
            plt.scatter(new_x_cen, new_y_cen, c='r', marker='o')
            plt.title("Iteration {}".format(ctr))
            plt.show()
        
        #Update the values
        x_cen = new_x_cen
        y_cen = new_y_cen
        smallest_dist_to_edge = np.min([x_cen, img.shape[1]-x_cen, y_cen, img.shape[0]-y_cen]) #smallest distance from centroid to edge of an image
        radius = np.min([0.9*radius, smallest_dist_to_edge]) #reduce radius by 0.75 each time
        radius = np.max([radius, 5])
        ctr += 1
        if ctr >= maxiter:
            #Maxmimum number of iterations reached
            break
        centroid_difference = np.sqrt((prev_cen_values['x'] - x_cen)**2 + (prev_cen_values['y'] - y_cen)**2)
        if centroid_difference < cen_tol:
            #Centroid has been unchanged within the tolerance
            break
        
        prev_cen_values['x'] = x_cen
        prev_cen_values['y'] = y_cen
         
    return x_cen, y_cen

def crop_image_for_fit(img, x_cen, y_cen, radius_scale=4):
    """
    Crop the image about the centroid for fitting
    INPUTS:
        ::np.ndaray:: img
        ::float:: x_cen
        ::float:: y_cen
        ::float:: radius_scale
    OUTPUTS
        ::np.ndarray:: <cropped_image>
        ::float:: crop_radius
    """
    img_max = np.max(img) #maximum pixel value of the image
    mean = np.mean(img) #mean value fo the image
    std = np.std(img) #standard deviation of the image
    
    #Threshold the image:
    #All pixels with values 4 standard deviations greater than the image mean
    #will have a value of img_max
    ret, thresh = cv.threshold(img, mean+4*std, img_max, cv.THRESH_BINARY)
    
    radius = np.sqrt(np.sum(thresh)/img_max/np.pi) #assume that the beam is circular
    crop_radius = radius * radius_scale
    
    #Find the smallest distance from the centroid to an edge of the image
    smallest_dist_to_edge = np.min([x_cen, img.shape[1]-x_cen, y_cen, img.shape[0]-y_cen])
    crop_radius = np.min([smallest_dist_to_edge, crop_radius]) #take the smaller of crop_radius and smallest_dist_to_edge
    return crop_image_safe(img, x_cen, y_cen, zoom_size=crop_radius), crop_radius

#######################################################################################
#######################################################################################


def slice_gui(img_og, img, Z_gaussian, x_cen, y_cen, crop_radius):
    """
    GUI for visualizing image
    """
    h, w = img.shape #height and width of image
    
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10,4), constrained_layout=True)
    #fig.tight_layout()
    im = ax[0].imshow(img_og, cmap='gray')
    #fig.colorbar(im, ax=ax[0], cmap='gray')
    rect = patches.Rectangle((x_cen-crop_radius, y_cen-crop_radius), crop_radius*2, crop_radius*2, linewidth=1, edgecolor='r', facecolor='none')
    ax[0].add_patch(rect)
    ax[0].set_title('Original Image')
    
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax[1])
    top_ax = divider.append_axes("top", size=1, pad=0.1, sharex=ax[1]) #Note: size and pad should be axes_grid.axes_size compatible
    top_ax.xaxis.set_tick_params(labelbottom=False) #don't show xlabel
    right_ax = divider.append_axes("right", size=1, pad=0.1, sharey=ax[1])
    right_ax.yaxis.set_tick_params(labelleft=False)
    
    top_ax.set_ylabel('Counts')
    right_ax.set_xlabel('Counts')
    
    #ax[1].set_xlabel('Horizontal Pixel Number')
    #ax[1].set_ylabel('Vertical Pixel Number')
    
    ax[1].imshow(img, cmap='gray')
    #ax[1].set_title('Cropped Region')
    
    ax[1].autoscale(enable=False)
    top_ax.autoscale(enable=False)
    right_ax.autoscale(enable=False)
    top_ax.set_ylim(top=np.max(img)*1.1)
    right_ax.set_xlim(right=np.max(img)*1.1)
    h_line = ax[1].axhline(np.nan, color='r')
    h_prof, = top_ax.plot(np.arange(w),np.zeros(w), 'r-')
    v_line = ax[1].axvline(np.nan, color='g')
    v_prof, = right_ax.plot(np.zeros(h), np.arange(h), 'g-')

    ax[2].imshow(Z_gaussian, cmap='gray')
    ax[2].set_title('Gaussian Fit of Cropped Region')

    def on_move(event):
        if event.inaxes is ax[1]:
            cur_x = event.xdata
            cur_y = event.ydata
    
            v_line.set_xdata([cur_x,cur_x])
            h_line.set_ydata([cur_y,cur_y])
            v_prof.set_xdata(img[:,int(cur_x)])
            h_prof.set_ydata(img[int(cur_y),:])
    
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect('motion_notify_event', on_move)
    
    plt.show()
    
if __name__ == "__main__":
    filename = "220214_1035_Image.jpg"
    #filename = "test2.png"
    img = cv.imread(filename, cv.IMREAD_GRAYSCALE)
    
    #from astropy.visualization import ZScaleInterval
    #interval = ZScaleInterval()
    #stretched_img = interval(img)
    #img = stretched_img
    plt.figure()
    plt.imshow(img)
    plt.colorbar()
    plt.show()
    
    '''
    x_cen, y_cen = get_image_centroid_vectorized(img)
    img_cropped = crop_image(img, x_cen, y_cen, zoom_size=50)
    plt.figure()
    plt.imshow(img_cropped)
    plt.colorbar()
    #plt.scatter(x_cen, y_cen, c='r', marker='o', s=50,)
    plt.show()
    '''
    
    x_cen, y_cen = find_centroid_iter(img, cen_tol=5, maxiter=10, do_plot=True)
    cropped_img, crop_radius = crop_image_for_fit(img, x_cen, y_cen, radius_scale=4)
    plt.figure()
    plt.imshow(cropped_img)
    plt.colorbar()
    plt.show()
    
    params, params_uncert, params_dict, params_uncert_dict, fit_fcn = fit_gaussian(cropped_img)
    x = np.arange(cropped_img.shape[1])
    y = np.arange(cropped_img.shape[0])
    X, Y = np.meshgrid(x, y)
    Z = gaussian(X, Y, *params)
    
    plt.figure()
    plt.imshow(Z)
    plt.colorbar()
    plt.show()
    
    slice_gui(img, cropped_img, Z, x_cen, y_cen, crop_radius)