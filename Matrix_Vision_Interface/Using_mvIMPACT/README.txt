April-June 2022
Matthew Leung

The scripts in this directory use the Matrix Vision mvIMPACT Acquire Python SDK to interface with the camera.
IMPORTANT: Instructions for setting up mvIMPACT Acquire and installing the API can be found in the G-CLEF
Fiber Lab Google Drive folder.
For general build instructions, look at:
https://www.matrix-vision.com/manuals/SDK_PYTHON/Building_page.html
The latest version of mvIMPACT Acquire can be found here:
http://static.matrix-vision.com/mvIMPACT_Acquire/

The code in this directory requires certain Python libraries to be installed. It is recommended to create a 
new conda environment for this code, if one doesn't exist already. See the Requirements folder for TXT files
which can be used to make a new conda environment with all the necessary requirements.

****************************************************************************************************************
*                                              IMPORTANT USAGE NOTE                                            *
****************************************************************************************************************
* BEFORE RUNNING THE SCRIPTS, MAKE SURE THAT YOU DID NOT ACCESS THE CAMERA VIA THE                             *
* MATRIX VISION WXPROPVIEW PROGRAM! IF YOU ACCESSED/CONTROLLED THE CAMERA USING THE WXPROPVIEW PROGRAM,        *
* THE SETTINGS WILL BE FIXED, AND THE SCRIPTS WILL NOT BE ABLE TO CHANGE THE SETTINGS (E.G. EXPOSURE TIME,     *
* BIT RATE). IF YOU DID USE THE WXPROPVIEW PROGRAM, UNPLUG THE CAMERA AND PLUG IT BACK IN FIRST BEFORE         *
* RUNNING ANY OF THE SCRIPTS.                                                                                  *
****************************************************************************************************************


------------------------------------------------------------------------------------------------------------------

WHAT EACH SCRIPT DOES:

mv_acquire_continuous.py acquires images from the Matrix Vision camera, and if desired, stacks them together.
With the camera turned on, run this script to continuously acquire images, and then stack them. This script
does not save results. Use this script as a tool to see if your exposure, gain, etc. settings are correct.
Then run mv_acquire_continuous_save.py or mv_acquire_continuous_save_loop.py to actually acquire and save images.

mv_acquire_continuous_save.py is the same as mv_acquire_continuous.py, but saves results in directory.

mv_acquire_continuous_save_loop.py is a looped version of mv_acquire_continuous_save.py. This script calls the
function acquire_continuous_and_stack from mv_acquire_continuous_save.py in a loop. This script was extensively
used for data acquisition during the mode scrambler tests.

------------------------------------------------------------------------------------------------------------------

WHAT IS SAVED:

Each stacked image is saved as a 12 bit TIF file (in 16 bit format) in a selected directory. Some image metadata
is saved in a JSON file with the same file basename as the image file. Examples of metadata include date and time
taken, exposure time, number of stacked frames, and actual FPS.
