# G-CLEF Fiber Lab
This repository contains code for the <b>[G-CLEF](https://gclef.cfa.harvard.edu/) Fiber Research Laboratory</b> at the [Center for Astrophysics | Harvard & Smithsonian (CfA)](https://cfa.harvard.edu/).

## Introduction

During the 2021-2022 school year, I completed a co-op/gap year research internship at the [Center for Astrophysics | Harvard & Smithsonian (CfA)](https://cfa.harvard.edu/), where I worked on [G-CLEF (GMT-Consortium Large Earth Finder)](https://gclef.cfa.harvard.edu/), a precision [radial velocity (RV)](https://en.wikipedia.org/wiki/Doppler_spectroscopy) echelle spectrograph which will be the first light instrument for the [Giant Magellan Telescope (GMT)](https://giantmagellan.org/). G-CLEF will search for and characterize Earth-like exoplanets.

The project I worked on within G-CLEF was a prototype <b>optical fiber mode scrambler</b> which agitates the fibers feeding the spectrograph from the GMT, in order to mitigate a phenomenon called modal noise which is detrimental to exoplanet detection. There are two parts to this project:
1) Designing and building a Fiber Characterization Station (FCS) which is used to determine certain metrics of optical fibers
2) Designing and building the optical fiber mode scrambler and then testing it using the FCS

<b>[Click here](https://github.com/mattleung10/G-CLEF_Fiber_Lab/blob/master/Summary/Matthew_Leung_CfA_Internship_Final_Report_V1.pdf) to read a report which explains my work in designing and building the FCS and mode scrambler.</b>

## Fiber Characterization Station (FCS)

### Design

The FCS allows for light to be injected into a test fiber, which is agitated by the mode scrambler. The FCS has four goals, which are to:
1) Image fiber input face (to ensure proper light injection location and alignment)
2) Image fiber near field (fiber output face)
3) Image fiber far field (collimated output of fiber)
4) Measure fiber output power and fiber focal ratio degradation (FRD)

The FCS design consists of four arms:
1) Pre-injection arm, to inject light into the fiber
2) Injection imaging arm, to image the fiber input face
3) Near field arm, to image the fiber near field
4) Far field arm, to image the fiber far field

<p float="center">
  <img src="https://github.com/mattleung10/G-CLEF_Fiber_Lab/blob/master/Summary/Images/FCS_overall_labelled.png" width="49%" />
  <img src="https://github.com/mattleung10/G-CLEF_Fiber_Lab/blob/master/Summary/Images/FCS_View.jpg" width="49%" />
</p>

### Camera Control Software

Three cameras are used in the FCS. A Pixelink PL-B781U CMOS camera is used for the injection imaging arm, and two Matrix Vision mvBlueCOUGAR-X102kG CMOS cameras are used for the near field arm and far field arm. Custom camera control code was written to control the cameras. The Pixelink camera is controlled using the script ```pixelink_class.py``` (in the ```Pixelink_Interface``` directory), which is a class which wraps the routines in the [pixelinkWrapper Python library](https://github.com/pixelink-support/pixelinkPythonWrapper). The Matrix Vision cameras are controlled using the mvIMPACT Python library (using the [mvIMPACT Acquire Python SDK](https://www.matrix-vision.com/manuals/SDK_PYTHON/)) and [Harvester Python library](https://github.com/genicam/harvesters). For more details, see the ```README.txt``` file inside the ```Matrix_Vision_Interface``` directory. Note that the Matrix Vision cameras acquire images in 12 bit, but the final images are saved as a 16 bit file.

Custom GUI tools were developed to help with the optical alignment progress:
1) GUI for aligning the injection imaging arm (```pixelink_class_gaussian_fit.py``` in the ```Pixelink_Interface``` directory)
2) GUI for aligning the near field arm (```mv_harvester_class_gaussian_fit_crop_GUI_pyqtgraph.py``` in the ```Matrix_Vision_Interface/Using_Harvester``` directory)
3) GUI for aligning the far field arm (```mv_harvester_class_ellipse_fit_pyqtgraph.py``` in the ```Matrix_Vision_Interface/Using_Harvester``` directory)

The latter two GUIs were created using the [PyQtGraph Python library](https://www.pyqtgraph.org/), and are shown below.
<p align="center">
  <img src="https://github.com/mattleung10/G-CLEF_Fiber_Lab/blob/master/Matrix_Vision_Interface/Using_Harvester/GUI_Demo/20220316_Near_Field_GUI_Demo.png" width="49%" />
  <img src="https://github.com/mattleung10/G-CLEF_Fiber_Lab/blob/master/Matrix_Vision_Interface/Using_Harvester/GUI_Demo/20220315_Far_Field_GUI_Demo.png" width="49%" />
</p>

Note that due to size constraints, images taken by the cameras were not included in this repository.

## Optical Fiber Mode Scrambler

### Background on Modal Noise

When coherent light propagates through a multi-mode optical fiber, the modes interfere at the fiber exit boundary, producing a high contrast speckle interference pattern called modal noise. This non-uniform interference pattern is a problem which particularly affects fiber-fed precision RV spectrographs like G-CLEF, leading to systematic errors and lower signal-to-noise ratios in measurements. To mitigate the effects of modal noise, a device called an optical fiber mode scrambler is used. A mode scrambler dynamically agitates a fiber, so that the interference pattern will change over time and be smoothed out over long exposures, destroying the modal information in the fiber.

Most fiber mode scramblers mechanically agitate the fiber in order to mitigate modal noise. However, the generality of mode scramblers in the literature are limited, because they are often specific to a given instrument. Different mode scramblers in the literature mechanically agitate a fiber in different ways, for example by shaking, bending, rotating, or twisting. However, there has not been much research into the best way to agitate a fiber so that the highest amount of modal noise is reduced. My internship project aimed to investigate this.

### Design

In this project, the mechanical design of the mode scrambler prototype consists of a [four-bar linkage](https://en.wikipedia.org/wiki/Four-bar_linkage) crank-rocker. Two stepper motors are used, which are controlled by an Arduino (see the ```Arduino_Code``` directory). An electrical schematic is shown below. For more details, please see the report linked in the Introduction.
<p align="center">
  <img src="https://github.com/mattleung10/G-CLEF_Fiber_Lab/blob/master/Summary/Images/Mode_Scrambler_Prototype_V1.2-1.png" width="50%" />
</p>

### Analysis Code and Mode Scrambler Testing

Custom software was written to analyze the images taken by the cameras described above, when the mode scrambler was ON and OFF. To test the mode scrambler, images of the fiber near field were analyzed, and the signal-to-noise ratio (SNR) of the pixels inside the fiber boundary was used as a metric for mode scrambling (for more details, please see the report linked in the Introduction). Various experiments were conducted, in which a particular parameter was varied (e.g. mode scrambler motor frequency) and its effect on SNR was investigated.

Images of the near field are considered as "science frames". In addition to these, dark frames, bias frames, and flat field frames were also taken. After obtaining a corrected image (dark and bias subtracted, and flat field corrected), the main goal is to identify the boundary of the test fiber's face in the corrected image. Then SNR and other metrics can be computed from the pixels inside the fiber face boundary. The process for identifying the boundary of the fiber face is non-trivial, because the mode scrambler was tested on fibers of different shapes (e.g. circular, square, octagonal, rectangular). To find the boundary of the fiber face, the [alpha shape](https://graphics.stanford.edu/courses/cs268-11-spring/handouts/AlphaShapes/as_fisher.pdf) was found. The process is summarized below:

1) Apply [Canny edge detector](https://en.wikipedia.org/wiki/Canny_edge_detector) to an 8 bit version of the corrected image, and obtain a binary image which represents the Canny edges. Let <i>S</i> be the set of nonzero points in the binary image.
2) Compute the [Delaunay Triangulation](https://en.wikipedia.org/wiki/Delaunay_triangulation) of <i>S</i>, called DT(<i>S</i>). Using DT(<i>S</i>), find the alpha complex <i>C</i><sub>α</sub>(<i>S</i>).
3) Find the boundary of the alpha shape, which is the boundary of <i>C</i><sub>α</sub>(<i>S</i>).
4) In some cases, the alpha shape can have an inner and an outer boundary. Take the outer boundary to be the boundary of the fiber face.
5) If desired, offset the boundary.
<p align="center">
  <img src="https://github.com/mattleung10/G-CLEF_Fiber_Lab/blob/master/Summary/Images/Alpha_Shape_Demo.png" width="75%" />
</p>

The final prototype mode scrambler was able to reduce modal noise by a factor of ~8. Near field images of an octagonal and rectangular test fiber, taken when the mode scrambler was ON and OFF, are shown below. Turning on the mode scrambler clearly reduces the modal noise. 
<p align="center">
  <img src="https://github.com/mattleung10/G-CLEF_Fiber_Lab/blob/master/Summary/Images/Oct_Fiber.png" width="75%" />
</p>
<p align="center">
  <img src="https://github.com/mattleung10/G-CLEF_Fiber_Lab/blob/master/Summary/Images/Rect_Fiber.png" width="75%" />
</p>

## Repository Directories

The following list explains what is inside each directory of this repository. All code was written in Python 3 (for the specific Python version, see the ```README.txt``` file inside each directory), with the exception of the code in the ```Arduino_Code``` directory, which contains Arduino sketches.
- ```Arduino_Code``` contains Arduino sketches to control the motors in the mode scrambler prototype.
- ```Gaussian_Fit``` contains code to fit a 2D Gaussian to an image. The code in this directory is used by scripts in other directories to focus the cameras.
- ```Matrix_Vision_Interface``` contains code to interface with the Matrix Vision mvBlueCOUGAR-X102kG cameras, which are used for the near field and far field arms. Use the code in this directory to focus the cameras, and to acquire data for the mode scrambler experiments.
- ```Near_Field_Analysis``` contains code to analyze near field images of fibers. This code is used to determine the boundary of the fiber face in each image, and some image metrics are computed from this.
- ```Pixelink_Interface``` contains code to interface with the Pixelink camera used in the injection arm. Use this code to focus the camera.
- ```Other``` contains miscellaneous code, mostly for testing/exploratory purposes.
- ```Summary``` contains a report summarizing my work in this internship, and some images for this README file.

Specific requirements can be found in TXT files inside each directory.
