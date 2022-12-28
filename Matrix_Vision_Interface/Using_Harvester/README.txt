March 2022
Matthew Leung

These scripts use the Harvester Python library to communicate with camera.
near_field_edge_contours.py contains functions to fit an ellipse to the fiber image.

--------------------------------------------------------------------------------------------

WHAT EACH SCRIPT DOES:

Gaussian_Fit.py is from the Gaussian_Fit directory. Fits an Gaussian function to image.

matrixvision_harvester_class.py is a class which can be instantiated to control the Matrix
Vision mvBlueCOUGAR-X102kG camera (or any other GenICam compliant camera) using the
Python Harvester library: https://github.com/genicam/harvesters

mv_harvester_class_ellipse_fit_pyqtgraph.py is used to align and focus the far field arm.
This script uses pyqtgraph to make a Qt GUI.
Finds the eccentricity of the ellipse which represents the far field fiber image, and plots
this in realtime.
For use with circular fibers only.

mv_harvester_class_gaussian_fit_crop_GUI_pyqtgraph.py is used to focus any arm containing
the Matrix Vision mvBlueCOUGAR-X102kG camera. This script uses pyqtgraph to make a Qt GUI.
The user can select a region of interest to fit a 2D Gaussian function to.
The GUI plots average FWHM (of the fitted Gaussian) as a function of time.

near_field_edge_contours.py contains functions used by mv_harvester_class_ellipse_fit_pyqtgraph.py
to find the contour of a fiber face, and to fit an ellipse to it.

--------------------------------------------------------------------------------------------

DIRECTORIES:

The directory OpenCV_Matplotlib_Display contains similar scripts, except that the GUIs are
created using OpenCV and/or Matplotlib interactive mode instead of pyqtgraph.

The directory Harvester_Test_Files contains scripts which were used for testing/exploratory
purposes.

The directory GUI_Demo contains images of the Qt GUIs created by pyqtgraph, with the scripts
mv_harvester_class_ellipse_fit_pyqtgraph.py and mv_harvester_class_gaussian_fit_crop_GUI_pyqtgraph.py
