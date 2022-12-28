#Matthew Leung
#Code last modified: February 28 2022
"""
Exploratory code to fit 1D Gaussian to image slice, instead of 2D Gaussian
to entire image. Also investigated the use of a clipped Gaussian, with a flat
top, in order to model oversaturated pixels.
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

def gaussian_1D(x, x0, sigma_x, A, back):
    return A * np.exp(-(x-x0)**2/(2*sigma_x**2)) + back

def gaussian_1D_clipped(x, x0, sigma_x, A, back, clip_lim):
    g = gaussian_1D(x, x0, sigma_x, A, back)
    return np.piecewise(x, [g < clip_lim, g >= clip_lim], [lambda xv:gaussian_1D(xv, x0, sigma_x, A, back), lambda xv:clip_lim])

def gaussian_FWHM(sigma):
    return 2*sigma*np.sqrt(2*np.log(2))

def gaussian_intensity(A, sigma_x, sigma_y, A_err, sigma_x_err, sigma_y_err):
    I = 2*np.pi * A * sigma_x * sigma_y
    I_err = I * np.sqrt((A_err/A)**2 + (sigma_x_err/sigma_x)**2 + (sigma_y_err/sigma_y)**2)
    return I, I_err

#######################################################################################
#######################################################################################
    
def fit_curve(x_data, y_data, y_sigma, fit_fcn, p0, bounds=None, maxfev=10000):
    """
    Fits curve using scipy.optimize.curve_fit. Code from APL.
    """
    ###########################################################################
    #Find the number of fit parameters in function and also the dof
    #See: https://stackoverflow.com/questions/847936/how-can-i-find-the-number-of-arguments-of-a-python-function/41188411#41188411
    from inspect import signature
    sig = signature(fit_fcn)
    dof = x_data.size - (len(sig.parameters) - 1)
    ###########################################################################
    
    if bounds is None:
        p, cov = scipy.optimize.curve_fit(fit_fcn, x_data, y_data, sigma=y_sigma, p0=p0, maxfev=maxfev)
    else:
        p, cov = scipy.optimize.curve_fit(fit_fcn, x_data, y_data, sigma=y_sigma, p0=p0, bounds=bounds, maxfev=maxfev)
    p_uncert = [] #store the uncertainty in the parameters
    # Calculate Chi-squared
    chisq = sum(((y_data-fit_fcn(x_data,*p))/y_sigma)**2)
    cdf = scipy.special.chdtrc(dof,chisq)
    # Convert Scipy cov matrix to standard covariance matrix.
    cov = cov*dof/chisq
    for i in range(len(p)) :
        p_uncert += [cov[i,i]**0.5*max(1,np.sqrt(chisq/dof))]
    cdf = scipy.special.chdtrc(dof,chisq)
    
    chisq_per_dof = chisq/dof
    ###########################################################################
    
    return p, p_uncert, chisq_per_dof, cdf

def fit_gaussian_1D_to_image(img, cen_coord):
    h, w = img.shape #height and width of image
    fit_fcn = gaussian_1D_clipped
    
    hor_x = np.arange(w)
    vert_x = np.arange(h)
    hor_slice = img[int(cen_coord[1]),:]
    vert_slice = img[:,int(cen_coord[0])]
    
    plt.figure()
    plt.plot(hor_x, hor_slice)
    plt.title('Horizontal Slice OG')
    plt.show()
    
    
    #gaussian_1D(x, x0, sigma_x, A, back)
    bounds = ((0, 1, 0, 0, 250), (w, w, np.inf, np.inf, 255))
    p0 = [w/2, w/5, np.max(img), np.max(img)/10, 254]    
    h_sigma = np.sqrt(hor_slice)
    h_sigma[h_sigma == 0] = 1
    v_sigma = np.sqrt(vert_slice)
    v_sigma[v_sigma == 0] = 1
    
    h_p, h_p_uncert, _, _ = fit_curve(hor_x, hor_slice, h_sigma, fit_fcn, p0, bounds=bounds, maxfev=10000)
    v_p, v_p_uncert, _, _ = fit_curve(vert_x, vert_slice, v_sigma, fit_fcn, p0, bounds=bounds, maxfev=10000)
    
    plt.figure()
    plt.plot(hor_x, fit_fcn(hor_x, *h_p))
    plt.title('Horizontal Slice')
    plt.show()
    
    plt.figure()
    plt.plot(vert_x, fit_fcn(vert_x, *h_p))
    plt.title('Vertical Slice')
    plt.show()
    
if __name__ == "__main__":
    hor_x = np.arange(100)
    plt.figure()
    plt.plot(hor_x, gaussian_1D_clipped(hor_x, 50, 10, 300, 10, 255))
    plt.show()
    