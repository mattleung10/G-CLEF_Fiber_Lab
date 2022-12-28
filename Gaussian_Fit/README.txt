February 2022
Matthew Leung

The scripts in this directory are used to fit a 2D Gaussian to an image.
This is used for alignment purposes, to focus optics.
A 2D Gaussian is fitted to the spot in the image, and the average FWHM of the fitted Gaussian
is used as a metric for focus.

--------------------------------------------------------------------------------------------------

WHAT EACH SCRIPT DOES:

Gaussian_Fit.py contains all the functions for the 2D Gaussian fit.
Usage Pipeline (can skip steps 1 and 2 if you have good way to locate the spot in the image):
    1) Iteratively find centroid of the frame/image by iteratively zooming into image and cropping
       to a smaller region (find_centroid_iter)
    2) Crop the image about the centroid (crop_image_for_fit)
    3) Fit a 2D Gaussian function to the cropped image (fit_gaussian)

Gaussian_Fit_Camera.py acquires images using OpenCV (using e.g. your computer's webcam), and
applies the functions in Gaussian_Fit.py. In the real world, the OpenCV routine to acquire image
frames should be replaced with the suitable function from your corresponding camera's Python API.
The results are plotted in realtime using matplotlib interactive mode.

Gaussian_Fit_Camera_Time.py is the same as Gaussian_Fit_Camera.py, except that a realtime plot of
time VS average FWHM (of the fitted Gaussian) is also displayed. This is what will eventually be
used, except again the OpenCV acquisition routine will need to be replaced.

--------------------------------------------------------------------------------------------------

The directory Gaussian_Plot_PDF contains a PDF file which can be printed and used to test
Gaussian_Fit_Camera.py and Gaussian_Fit_Camera_Time.py. This PDF file contains a Gaussian PSF
plotted with an inverted colormap (white background, Gaussian is black).
Hold out the printed paper in front of your computer webcam and see what the resulting fitted
Gaussian PSF looks like.
