February 2022
Matthew Leung

The code in this directory is used to interface with the Pixelink camera.

--------------------------------------------------------------------------------------------------

WHAT EACH SCRIPT DOES:

pixelink_class.py is a class that wraps the routines in the pixelinkWrapper Python library.
Create an instance of this class in order to control the Pixelink camera, which is used to image
the fiber injection in the G-CLEF Fiber Lab Fiber Characterization Station (FCS).

pixelink_class_gaussian_fit.py acquires images using the Pixelink camera in a loop.
For each image, using the functions in Gaussian_Fit.py, this script:
    1) Iteratively finds centroid of the frame/image (find_centroid_iter)
    2) Crops the image about the centroid (crop_image_for_fit)
    3) Fits a 2D Gaussian function to the cropped image (fit_gaussian)
    4) Finds the average FWHM of the fitted Gaussian
It then plots the results in realtime, and plots time VS average FWHM.
THIS IS THE SCRIPT YOU WANT TO RUN IF YOU WANT TO FOCUS THE INJECTION ARM.

pixelink_class_test.py is just a test of pixelink_class.py. Use this to see if the camera and its
connection are working.
