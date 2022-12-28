April 2022
Matthew Leung

This directory contains code which is used to interface with the Matrix Vision mvBlueCOUGAR-X102kG cameras,
which are used for the near field and far field arms.

It is best to use the Matrix Vision mvIMPACT Python library to control the cameras, because when using this,
it was confirmed that the bit rate, exposure time, gain, and frame rate (to a certain extent, limited by MTU)
can be properly adjusted.

--------------------------------------------------------------------------------------------------------------

DIRECTORIES:

The directory Using_Harvester uses the Harvester Python library. The Harvester library should work with all
GenICam compliant cameras. The GitHub repository can be found here:
https://github.com/genicam/harvesters

The directory Using_mvIMPACT uses the Matrix Vision mvIMPACT Acquire Python SDK to interface with the camera.
IMPORTANT: Instructions for setting up mvIMPACT Acquire and installing the API can be found in the G-CLEF
Fiber Lab Google Drive folder.
For general build instructions, look at:
https://www.matrix-vision.com/manuals/SDK_PYTHON/Building_page.html
The latest version of mvIMPACT Acquire can be found here:
http://static.matrix-vision.com/mvIMPACT_Acquire/
