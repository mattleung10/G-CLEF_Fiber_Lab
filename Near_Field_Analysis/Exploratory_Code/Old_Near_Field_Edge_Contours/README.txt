March 2022
Matthew Leung

These scripts attempt to find the contour bounding the face fo a circular fiber.

The final script which you should use is near_field_edge_contours.py
Procedure:
    1) Gaussian blur the image to smooth out noise
    2) Threshold the image using Otsu's Method
    3) Find contours of thresholded image
    4) Fit an ellipse to each contour
    5) Find the eccentricity and area of each ellipse
    6) Find the ellipse with the largest area, and take that as the ellipse
       which encloses the circular fiber face

THIS PROCEDURE IS OLD AND ONLY WORKS WELL FOR NEAR FIELD CIRCULAR FIBERS.
FOR FAR FIELD, ONLY USE IT TO DETERMINE THE ECCENTRICITY, AND NOT TO FIND THE
CONTOUR BOUNDING THE FIBER IMAGE.
