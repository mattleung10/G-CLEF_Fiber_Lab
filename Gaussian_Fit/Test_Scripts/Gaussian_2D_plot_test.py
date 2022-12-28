#Matthew Leung
#February 10, 2022
"""
Plots a 2D Gaussian
"""

import numpy as np
import matplotlib.pyplot as plt

def gaussian_2D(x, y, x0, y0, xalpha, yalpha, A):
    return A * np.exp(-(x-x0)**2/(2*xalpha**2) - (y-y0)**2/(2*yalpha**2))

def plot_2D_gaussian():
    x = np.linspace(-4, 4, num=500)
    y = np.linspace(-4, 4, num=500)
    X, Y = np.meshgrid(x, y)
    Z = gaussian_2D(X, Y, x0=0, y0=0, xalpha=2, yalpha=2, A=4)
    plt.figure(figsize=(8,6), dpi=100)
    plt.imshow(Z)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Intensity')
    plt.title('2D Gaussian')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show() 

if __name__ == "__main__":
    plot_2D_gaussian()
