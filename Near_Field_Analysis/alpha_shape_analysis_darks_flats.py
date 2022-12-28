#Matthew Leung
#June 2022
"""
This script contains driver code for functions from alpha_shape_id_roi.py.
Use this script on a directory of images to find the alpha shape of the fiber
face in each image.

This script allows for subtraction of dark and bias frames.
Version 2: Proper uncertainty calculation
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import pandas as pd
import glob
import json
import copy
import time
from alpha_shape_id_roi import hull_mask_from_alpha_shape, read_12bit_image_saved_as_16bit

#######################################################################################
#######################################################################################

def load_json(json_filename):
    #Loads a JSON file with filename json_filename into a dictionary
    with open(json_filename, 'r') as f: #open the json file and load into a dict
        json_dict = json.load(f) #this is a dict
    return json_dict

def save_json(json_savename, json_dict):
    #Saves a dictionary json_dict as a JSON file, with filename json_savename
    with open(json_savename, 'w', encoding='utf-8') as f:
        json.dump(json_dict, f, ensure_ascii=False, indent=4)
    return True

def get_json_dict_list(images_filenames_list):
    json_dict_list = [] #List of dict
    #Iterate over all image filenames in images_filenames_list and get their corresponding JSON files
    for filename in images_filenames_list:
        #Load the json file which is associated with the image
        json_filename = os.path.splitext(filename)[0] + ".json" #the filename of the associated json file to the image
        json_dict = load_json(json_filename)
        json_dict["image_filename"] = filename
        json_dict_list += [json_dict]
    return json_dict_list

#######################################################################################
#######################################################################################

def stack_images(img_list, shape=None):
    """
    Stack images in a list, and normalize the final image
    """
    if shape is None:
        shape = np.shape(img_list[0])
        
    stacked_arr = np.zeros(shape=shape)
    for i in range(0,len(img_list),1):
        curr_img = img_list[i]
        stacked_arr += curr_img
    
    #Normalize the stacked iamge
    stacked_arr_norm = stacked_arr / len(img_list)
    return stacked_arr_norm

def process_darks(darks_dir, dark_message="Dark"):
    """
    Returns a dictionary where each key is an exposure time value in us,
    and the value corresponding to that key is a stacked dark frame.
    INPUTS:
        ::str:: darks_dir    #directory storing the dark frames
        ::str:: dark_message #the str in the message field of the JSON file which denotes a dark frame
    OUTPUT:
        ::dict:: dict_of_dark_images
    """
    search_str = darks_dir + "/*.tif"
    images_filenames_list = glob.glob(search_str) #list of full filenames of images in darks_dir
    
    json_dict_list = get_json_dict_list(images_filenames_list)
        
    df = pd.DataFrame(json_dict_list)
    df_darks = df[df["message"] == dark_message] #only select images which have a message <dark_message>
    df_darks = df_darks[df_darks["number of frames"] == 1] #only select single frames
    
    unique_exp_times = df_darks["exposure_time_us"].unique().tolist() #list of exposure times
    dict_of_dark_images = dict()
    for exp_time in unique_exp_times:
        df_exp = df_darks[df_darks["exposure_time_us"] == exp_time]
        curr_images_filenames_list = df_exp["image_filename"].tolist()
        
        img_list = []
        for filename in curr_images_filenames_list:
            img = read_12bit_image_saved_as_16bit(filename, verbosity=0) #read the image
            img_list += [img]
            
        stacked_dark = stack_images(img_list) #stack the dark frames
        dict_of_dark_images[exp_time] = stacked_dark
     
    return dict_of_dark_images

def get_dark_and_bias(exp_time, dict_of_dark_images, min_exp_time_us=10):
    if exp_time in dict_of_dark_images.keys():
        dark = dict_of_dark_images[exp_time]
    else:
        dark = None
    if min_exp_time_us in dict_of_dark_images.keys():
        bias = dict_of_dark_images[min_exp_time_us]
    else:
        bias = None
    return dark, bias

def get_flat(flats_dir, flat_message="Flat"):
    """
    Given a directory flats_dir containing flat field frames (all with the same
    exposure time), returns a single stacked image of all the flat field frames
    INPUTS:
        ::str:: flats_dir
        ::str:: flat_message
    OUTPUT:
        ::np.ndarray:: stacked_flat
        ::float:: exp_time
    """
    search_str = flats_dir + "/*.tif"
    images_filenames_list = glob.glob(search_str) #list of full filenames of images in images_dir
    
    json_dict_list = get_json_dict_list(images_filenames_list)
    df = pd.DataFrame(json_dict_list)
    df_flats = df[df["message"] == flat_message] #only select images which have a message <dark_message>
    df_flats = df_flats[df_flats["number of frames"] == 1] #only select single frames
    
    unique_exp_times = df_flats["exposure_time_us"].unique().tolist()
    if len(unique_exp_times) > 1:
        raise ValueError("Multiple exposure times for flat images")
        
    flat_filenames_list = df_flats['image_filename'].tolist()
    img_list = [read_12bit_image_saved_as_16bit(filename, verbosity=0) for filename in flat_filenames_list]
    stacked_flat = stack_images(img_list)
    
    exp_time = unique_exp_times[0]
    return stacked_flat, exp_time

#######################################################################################

def dark_bias_correction(img_og, exp_time, dict_of_dark_images, min_exp_time_us=10):
    """
    Subtracts dark and bias frames
    INPUTS:
        ::np.ndarray:: img_og        #original image
        ::float:: exp_time           #exposure time of original image in [us]
        ::dict:: dict_of_dark_images #return value of process_darks function
        ::float:: min_exp_time_us    #the exposure time in [us] for bias frames
    """
    img = img_og - dict_of_dark_images[exp_time]
    if min_exp_time_us in dict_of_dark_images.keys():
        img = img - dict_of_dark_images[min_exp_time_us]
    img = np.clip(img, 0, 4096) #clip values so there aren't any negative values or values above 2^12
    return img

def flat_dark_bias_correction(sci_img, t_sci, dark_img, t_dark, bias_img=None, flat_img=None, t_flat=None, clip_lim=4096):
    """
    Applies dark,bias, and flat field correction.
    See http://www.bu.edu/astronomy/wp-assets/script-files/buas/oldsite/astrophotography/flat.htm
    for more information about dark, bias, and flat field correction.
    INPUTS:
        ::np.ndarray:: sci_img  #science frame
        ::float:: t_sci         #science frame exposure time
        ::np.ndarray:: dark_img #dark frame
        ::float:: t_dark        #dark frame exposure time
        ::np.ndarray:: bias_img #bias frame
        ::np.ndarray:: flat_img #flat frame
        ::float:: t_flat        #flat frame exposure time
        ::int:: clip_lim        #upper limit to clip image
    OUTPUT:
        ::np.ndarray::
    """
    if bias_img is None:
        bias_img = np.zeros(shape=sci_img.shape)
    
    numerator = (sci_img - bias_img) - t_sci/t_dark * (dark_img - bias_img)
    if flat_img is None: #No flat field image provided
        return np.clip(numerator, 0, clip_lim) #clip values so there aren't any negative values or values above 2^12
    
    denominator = (flat_img - bias_img) - t_flat/t_dark * (dark_img - bias_img)
    flat_corrected = numerator / denominator
    final_corrected = flat_corrected * np.mean(denominator)
    final_corrected = np.clip(final_corrected, 0, clip_lim) #clip values so there aren't any negative values or values above 2^12
    return final_corrected
    

#######################################################################################
#######################################################################################
    
def compute_SNR(hull_mask, img):
    """
    Computes the signal-to-noise ratio of pixels within the hull mask
    INPUTS:
        ::np.ndarray:: img
        ::np.ndarray:: dark_img
    """
    whered = np.where(hull_mask==1, img, -1000) #-1000 can instead be any arbitrary number which is NOT a valid pixel value
    pixel_vals_in_hull = whered[np.where(whered != -1000)] #list of pixel values inside the mask
    N = np.size(pixel_vals_in_hull) #number of pixels in hull
    
    pixel_vals_in_hull_err = np.sqrt(pixel_vals_in_hull)

    mu = np.mean(pixel_vals_in_hull)
    sigma = np.std(pixel_vals_in_hull)
    
    mu_err = 1/N * np.sqrt(np.sum(pixel_vals_in_hull_err**2))
    term = np.power((pixel_vals_in_hull-mu)/(sigma*N) * pixel_vals_in_hull_err, 2)
    sigma_err = np.sqrt(np.sum(term))
    
    SNR = np.divide(mu, sigma)
    SNR_err = SNR * np.sqrt((mu_err/mu)**2 + (sigma_err/sigma)**2)
    return mu, mu_err, sigma, sigma_err, SNR, SNR_err

def driver(images_dir, flats_dir=None, save_boundary_points=False, shape_offset_amount=None):    
    """
    Driver code which applies bias, dark, and flat-field corrections to images
    of the near field fiber, and then analyzes the images. This function calls
    hull_mask_from_alpha_shape from alpha_shape_id_roi.py in order to find the
    boundary of the fiber face, and then computes some image metrics.
    
    It is assumed that there is a folder with dark frames, called "Darks",
    inside images_dir.
    
    INPUTS:
        ::str:: images_dir               #directory of images
        ::str:: flats_dir                #directory of flat field frames (optional)
        ::boolean:: save_boundary_points #whether or not to save the boundary points (default False)
        ::float:: shape_offset_amount    #number of pixels to offset the original alpha shape by
    """
    search_str = images_dir + "/*.tif"
    images_filenames_list = glob.glob(search_str) #list of full filenames of images in images_dir
    
    results_savedir = os.path.join(images_dir, "Results")
    if os.path.isdir(results_savedir) == False: os.mkdir(results_savedir)
    
    ###########################################################################
    #Get dark and bias frames
    min_exp_time_us = 10 #minimum exposure time in [us]
    darks_dir = os.path.join(images_dir, "Darks")
    if os.path.isdir(darks_dir):
        print("DARKS DIRECTORY: "+darks_dir)
        print("Processing darks...")
        dict_of_dark_images = process_darks(darks_dir)
        if min_exp_time_us in dict_of_dark_images.keys():
            print("Bias images found")
        print("Processing darks complete")
    else:
        print("DARKS DIRECTORY: None given")
    print("\n")
    ###########################################################################
    #Get flat field frame
    if flats_dir is not None:
        if os.path.isdir(flats_dir):
            print("FLATS DIRECTORY: "+flats_dir)
            flat_img, t_flat = get_flat(flats_dir)
            flat_img = np.clip(flat_img, 1, 4096) #if pixel has value of 0, dividing by flat will cause error
        else:
            print("The directory "+flat_img+ " is invalid.")
            flat_img = None
            t_flat = None   
    else:
        print("FLATS DIRECTORY: None given")
        flat_img = None
        t_flat = None
    ###########################################################################
    
    print("\nPROCESSING IMAGES...")
    st = time.time()
    #Iterate over all filenames in images_filenames_list
    for i in range(0,len(images_filenames_list),1):
        filename = images_filenames_list[i]
        basename = os.path.basename(filename) #file basename, with extension
        basename_no_ext = os.path.splitext(basename)[0] #file basename, without extension
        elapsed_time = time.time() - st
        print("Processing file {}/{} | ".format(i+1, len(images_filenames_list)) + basename+" | time elapsed = {:.3f} seconds = {:.3f} minutes".format(elapsed_time, elapsed_time/60.0))
        
        #Load the json file which is associated with the image
        json_filename = os.path.splitext(filename)[0] + ".json" #the filename of the associated json file to the image
        json_dict = load_json(json_filename)
        t_sci = json_dict["exposure_time_us"] #science frame exposure time
        
        sci_img = read_12bit_image_saved_as_16bit(filename, verbosity=0) #read the image
        dark_img, bias_img = get_dark_and_bias(t_sci, dict_of_dark_images, min_exp_time_us=min_exp_time_us)
        t_dark = t_sci #dark frame exposure time
        
        #Apply dark, bias, flat field correction to obtain final image, called img
        img = flat_dark_bias_correction(sci_img, t_sci, dark_img, t_dark, bias_img=bias_img, flat_img=flat_img, t_flat=t_flat)
        
        ###############################################################################
        #APPLY ALPHA SHAPES AND SAVE THE MASK
        #Find the hull mask, which includes all pixels inside the alpha shape boundary
        if save_boundary_points == True:
            hull_mask, boundary_points = hull_mask_from_alpha_shape(img, return_boundary_points=True, shape_offset_amount=shape_offset_amount, verbosity=0) 
        else:
            hull_mask = hull_mask_from_alpha_shape(img, return_boundary_points=False, shape_offset_amount=shape_offset_amount, verbosity=0) 
        #Save the mask; write it to an image
        hull_mask_savename = os.path.join(results_savedir, basename_no_ext+"_mask.bmp")
        cv.imwrite(hull_mask_savename, 255 * hull_mask)
        
        #Apply the mask to the original image
        whered = np.where(hull_mask==1, img, -1000) #-1000 can instead be any arbitrary number which is NOT a valid pixel value
        list_of_pixel_vals_in_hull = whered[np.where(whered != -1000)] #list of pixel values inside the mask
        
        ###############################################################################
        #COMPUTE STATISTICS AND SAVE RESULTS IN NEW JSON FILE
        
        mu, mu_err, sigma, sigma_err, SNR, SNR_err = compute_SNR(hull_mask, img)

        #Save results in dict; add onto a copy of the original json file
        json_dict['pixels_in_hull'] = len(list_of_pixel_vals_in_hull)
        json_dict['mean'] = mu
        json_dict['mean_err'] = mu_err
        json_dict['std'] = sigma
        json_dict['std_err'] = sigma_err
        json_dict['mean/std'] = SNR
        json_dict['mean/std_err'] = SNR_err

        #Save json file in results directory        
        json_savename = os.path.join(results_savedir, basename_no_ext+"_stats.json")
        save_json(json_savename, json_dict)
        
        #Save the boundary points (for plotting purposes)
        if save_boundary_points == True:
            boundary_savename = os.path.join(results_savedir, basename_no_ext+"_boundary.txt")
            np.savetxt(boundary_savename, boundary_points)
    
    elapsed_time = time.time() - st
    print("Image processing took {:.3f} seconds = {:.3f} minutes".format(elapsed_time, elapsed_time/60.0))
    return True

if __name__ == "__main__":
    #Image folder
    #CHANGE THIS LINE AS REQUIRED:
    images_dir = os.path.join(os.getcwd(), "Images", "20220602_Stack_Vary")
    images_dir = os.path.join(os.getcwd(), "Images", "20220602_Exp_Vary")
    images_dir = os.path.join(os.getcwd(), "Images", "20220603_Tot_Exp_Const")
    images_dir = os.path.join(os.getcwd(), "Images", "20220613_Dist_Vary")
    images_dir = os.path.join(os.getcwd(), "Images", "20220615_Dist_Vary_Fixed_Fiber")
    images_dir = os.path.join(os.getcwd(), "Images", "20220617_Twist")
    images_dir = os.path.join(os.getcwd(), "Images", "20220623_Freq_Vary")
    images_dir = os.path.join(os.getcwd(), "Images", "20220627_Freq_Vary")
    images_dir = os.path.join(os.getcwd(), "Images", "20220628_Freq_Vary_Oct")
    
    #Flats folder
    #If there is no flats folder, just do flats_dir = None
    #CHANGE THIS LINE AS REQUIRED:
    #flats_dir = os.path.join(os.getcwd(), "Images", "20220613_Flats_V2")
    #flats_dir = None
    flats_dir = os.path.join(os.getcwd(), "Images", "20220627_Freq_Vary", "Flats")
    
    driver(images_dir, flats_dir, save_boundary_points=True, shape_offset_amount=-3)
    
    