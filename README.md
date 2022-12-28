# G-CLEF Fiber Lab Code
This repository contains code for the <b>[G-CLEF](https://gclef.cfa.harvard.edu/) Fiber Research Laboratory</b> at the [Center for Astrophysics | Harvard & Smithsonian (CfA)](https://cfa.harvard.edu/).

## Introduction

During the 2021-2022 school year, I completed a co-op/gap year research internship at the [Center for Astrophysics | Harvard & Smithsonian (CfA)](https://cfa.harvard.edu/), where I worked on [G-CLEF (GMT-Consortium Large Earth Finder)](https://gclef.cfa.harvard.edu/), a precision [radial velocity](https://en.wikipedia.org/wiki/Doppler_spectroscopy) echelle spectrograph which will be the first light instrument for the [Giant Magellan Telescope (GMT)](https://giantmagellan.org/). G-CLEF will search for and characterize Earth-like exoplanets.

The project I worked on within G-CLEF was a prototype <b>optical fiber mode scrambler</b> which agitates the fibers feeding the spectrograph from the GMT, in order to mitigate a phenomenon called modal noise which is detrimental to exoplanet detection. There are two parts to this project:
1) Designing and building a Fiber Characterization Station (FCS) which is used to determine certain metrics of optical fibers
2) Designing and building the optical fiber mode scrambler and then testing it using the FCS

<b>[Click here](https://github.com/mattleung10/G-CLEF_Fiber_Lab/blob/master/Summary/Matthew_Leung_CfA_Internship_Final_Report_V1.pdf) to read a report which explains my work in designing and building the FCS and mode scrambler.</b>

## Fiber Characterization Station (FCS)

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

Three cameras were used. 

## Optical Fiber Mode Scrambler

TBD

## Repository Directories

The following list explains what is inside each directory of this repository. All code was written in Python 3, with the exception of the code in the ```Arduino_Code``` directory, which contains Arduino sketches.
- ```Arduino_Code``` contains Arduino sketches to control the motors in the mode scrambler prototype.
- ```Gaussian_Fit``` contains code to fit a 2D Gaussian to an image. The code in this directory is used by scripts in other directories to focus the cameras.
- ```Matrix_Vision_Interface``` contains code to interface with the Matrix Vision mvBlueCOUGAR-X102kG cameras, which are used for the near field and far field arms. Use the code in this directory to focus the cameras, and to acquire data for the mode scrambler experiments.
- ```Near_Field_Analysis``` contains code to analyze near field images of fibers. This code is used to determine the boundary of the fiber face in each image, and some image metrics are computed from this.
- ```Pixelink_Interface``` contains code to interface with the Pixelink camera used in the injection arm. Use this code to focus the camera.
- ```Other``` contains miscellaneous code, mostly for testing/exploratory purposes.
- ```Summary``` contains a report summarizing my work in this internship, and some images for this README file.

Specific requirements can be found in TXT files inside each directory.
