June 2022
Matthew Leung

This directory contains code to analyze near field images of fibers.

The requirements are found in the Requirements directory.

--------------------------------------------------------------------------------------------------

STEPS TO RUN THE SCRIPTS IN THIS DIRECTORY:
   1) Place the images you want to analyze into the directory "Images".
      All images of the same experiment / data collection trial should be put into the SAME
      subdirectory within "Images". If there are dark/bias frames, put them in a "Darks" folder
      under that subdirectory.
      E.g.: All data take on June 27, 2022, in which the frequency of the mode scrambler
      was varied, was placed in a subdirectory within "Images" called "20220627_Freq_Vary".
   2) In alpha_shape_analysis_darks_flats.py, specify the subdirectory of images you want to use.
      A "Flats" directory could also be specified. After the directories have been specified,
      run this script to find the boundary of the fiber face in each image in the subdirectory,
      and to compute image metrics. The results will be saved in a folder called "Results" under
      that subdirectory of images.
   3) Inside the directory "Results_Analysis", create a Python script to analyze the results from
      step 2. Run the script. Plots will be saved in a subdirectory of the corresponding
      "Results" folder.

Example: Suppose we take data on MMDDYYYY. Here's what the directories should look like:

STEP 1:

    Near_Field_Analysis
    ├── Images
    |   └── MMDDYYYY_experiment_name
    |       ├── Darks
    |       └── Flats                            <-- This is optional and can be in a
    |                                                different location
    ├── Results_Analysis
    ├── alpha_shape_analysis_darks_flats.py
    └── alpha_shape_id_roi.py

STEP 2:

    Near_Field_Analysis
    ├── Images
    |   └── MMDDYYYY_experiment_name
    |       ├── Darks
    |       └── Flats
    ├── Results_Analysis
    ├── alpha_shape_analysis_darks_flats.py     <-- In this script, specify location of 
    |                                               MMDDYYYY_experiment_name, then run this script
    └── alpha_shape_id_roi.py

AFTER RUNNING STEP 2:

    Near_Field_Analysis
    ├── Images
    |   └── MMDDYYYY_experiment_name
    |       ├── Darks
    |       ├── Flats
    |       └── Results                         <-- Results of step 2 will be saved in here
    ├── Results_Analysis
    ├── alpha_shape_analysis_darks_flats.py
    └── alpha_shape_id_roi.py

STEP 3:

    Near_Field_Analysis
    ├── Images
    |   └── MMDDYYYY_experiment_name
    |       ├── Darks
    |       ├── Flats
    |       └── Results                         
    ├── Results_Analysis
    |   └── MMDDYYYY_experiment_name.py          <-- Create a script to analyze the results 
    |                                                of MMDDYYYY_experiment_name
    ├── alpha_shape_analysis_darks_flats.py
    └── alpha_shape_id_roi.py

AFTER RUNNING STEP 3:

    Near_Field_Analysis
    ├── Images
    |   └── MMDDYYYY_experiment_name
    |       ├── Darks
    |       ├── Flats
    |       └── Results
    |           └── Plots                        <-- Plots will be saved in here
    ├── Results_Analysis
    |   └── MMDDYYYY_experiment_name.py
    ├── alpha_shape_analysis_darks_flats.py
    └── alpha_shape_id_roi.py


--------------------------------------------------------------------------------------------------

HOW NEAR-FIELD FIBER FACE BOUNDARY IS COMPUTED:
    1) Apply Canny edge detector to 8 bit version of image, and obtain a binary image
       which represents the Canny edges
    2) Take the nonzero points from the binary image, and find the alpha
       shape of these points
    3) Find the outer boundary of the alpha shape
    4) If desired, offset the alpha shape
    5) Create a binary mask of all pixels inside the alpha shape outer boundary

alpha_shape_id_roi.py contains the code to do this. See the hull_mask_from_alpha_shape function.

alpha_shape_analysis_darks_flats.py does bias, dark, and flat-field correction on images in a
subdirectory, and then calls hull_mask_from_alpha_shape in alpha_shape_id_roi.py on all the
corrected images.
