#Matthew Leung
#May 2022
#Python 3
"""
Looped version of mv_acquire_continuous_save.py
"""

import time
import os
from mv_acquire_continuous_save import acquire_continuous_and_stack, get_user_input, play_finished_sound

def acquire_continuous_and_stack_loop(num_pics, num_frames, exp_time_us, FPS, savedir, message=None):
    """
    Calls the acquire_continuous_and_stack function in mv_acquire_continuous_save.py
    multiple times.
    INPUTS:
        ::int:: num_pics      #number of pictures to take
        ::int:: num_frames    #number of stacked frames for each picture
        ::float:: exp_time_us #exposure time of each individual frame
        ::float:: FPS         #desired frame rate
        ::str:: savedir       #directory to save pictures
        ::str:: message       #message for the pictures (this is for metadata)
    """
    
    if message is None:
        message = get_user_input()
    
    global_st = time.time()
    for i in range(0,num_pics,1):
        iteration_st = time.time()
        print("\nAcquiring stacked image {} out of {}...".format(i+1, num_pics))
        acquire_continuous_and_stack(num_frames, exp_time_us, FPS, savedir, message=message)
        print("This iteration took {} seconds".format(time.time()-iteration_st))
        print("Total time elapsed is {} seconds".format(time.time()-global_st))
    return True


if __name__ == "__main__":
    #Directory to store the folders containing the images for each data collection run
    main_savedir = "Saved_Data"
    if os.path.isdir(main_savedir) == False: os.mkdir(main_savedir)
    #This is where the acquired images will be saved
    savedir = os.path.join(main_savedir, "mv_acquire_continuous_save_loop")
    if os.path.isdir(savedir) == False: os.mkdir(savedir)
    
    ###########################################################################
    #CHANGE THESE AS REQUIRED:
    num_frames = 1 #number of stacked frames
    exp_time_us = 10 #exposure time in microseconds
    FPS = 25 #desired frame rate
    num_pics = 10 #number of pictures to take
    ###########################################################################
    
    global_st = time.time()
    acquire_continuous_and_stack_loop(num_pics, num_frames, exp_time_us, FPS, savedir)
    global_et = time.time()
    print("Total time elapsed is {} seconds = {} minutes = {} hours".format(global_et-global_st, (global_et-global_st)/60, (global_et-global_st)/3600))
    
    play_finished_sound()
    