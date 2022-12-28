#Matthew Leung
#February 14, 2022
"""
This script plots a Gaussian PSF, and saves it as a PDF file.
Purpose: Creates a Gaussian PSF with an inverted colourmap, which can then
be printed out on a piece of paper for testing purposes.
"""

import os
import numpy as np
import matplotlib.pyplot as plt


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



if __name__ == "__main__":
    dim = 2000
    x = np.linspace(-4,4,num=dim)
    y = np.linspace(-4,4,num=dim)
    X, Y = np.meshgrid(x, y)
    Z = gaussian(X, Y, x0=0, y0=0, sigma_x=0.25, sigma_y=0.175, A=1, back=0, theta=0)
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,8), dpi=400)
    ims = ax.imshow(Z, cmap='Greys', origin='lower')
    ax.axis('off')
    plt.savefig("gaussian_example.pdf")
    plt.show()