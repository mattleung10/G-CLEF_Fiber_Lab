#Matthew Leung
#February 2022
"""
Just a script that I used to help understand a problem, to debug something
"""

import numpy as np
import matplotlib.pyplot as plt
from Gaussian_Fit import fit_gaussian, gaussian_clipped, gaussian_FW_at_val
from Gaussian_Fit_1D import fit_gaussian_1D_to_image

if __name__ == "__main__":
    cropped_img = np.loadtxt("error.txt")
    plt.figure()
    plt.imshow(cropped_img)
    plt.colorbar()
    plt.show()
    
    h, w = cropped_img.shape
    xpoints = np.arange(w)
    plt.figure()
    plt.plot(xpoints, cropped_img[int(w/2),:])
    plt.show()    

    
    params, params_uncert, params_dict, params_uncert_dict, fit_fcn = fit_gaussian(cropped_img, clipped_gaussian=True)
    A = params[4]
    sigma_x = params[2]
    print('Oversaturated width_x = {} pixels'.format(gaussian_FW_at_val(A, sigma_x, 255)))
    
    x = np.arange(0, w)
    y = np.arange(0, h)
    X, Y = np.meshgrid(x, y)
    Z = gaussian_clipped(X, Y, *params)
    
    plt.figure()
    plt.imshow(Z)
    plt.colorbar()
    plt.title('Fit')
    plt.show()
    
    residuals = cropped_img - Z
    plt.figure()
    plt.imshow(residuals)
    plt.colorbar()
    plt.title('Residuals')
    plt.show()
    
    #cen_coord = [w/2, h/2]
    #fit_gaussian_1D_to_image(cropped_img, cen_coord)
    
    #this is what went wrong: you did y_sigma = np.sqrt(img),
    #but if image values have 0, then the y_sigma vals will be 0, and chi squared calc will fail