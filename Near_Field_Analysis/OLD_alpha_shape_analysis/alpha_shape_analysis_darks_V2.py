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
from alpha_shape_id_roi import hull_mask_from_alpha_shape, read_12bit_image_saved_as_16bit

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
    
    json_dict_list = [] #List of dict
    #Iterate over all image filenames in images_filenames_list and get their corresponding JSON files
    for filename in images_filenames_list:
        #Load the json file which is associated with the image
        json_filename = os.path.splitext(filename)[0] + ".json" #the filename of the associated json file to the image
        json_dict = load_json(json_filename)
        json_dict["image_filename"] = filename
        json_dict_list += [json_dict]
        
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

def subtract_dark_and_bias(img_og, exp_time, dict_of_dark_images, min_exp_time_us=10):
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

#######################################################################################

def compute_SNR(hull_mask, img_og, dark_img=None, bias_img=None):
    """
    INPUTS:
        ::np.ndarray:: img
        ::np.ndarray:: dark_img
    """
    whered_sci = np.where(hull_mask==1, img_og, -1000) #-1000 can instead be any arbitrary number which is NOT a valid pixel value
    sci_pixel_vals_in_hull = whered_sci[np.where(whered_sci != -1000)] #list of pixel values inside the mask
    N = np.size(sci_pixel_vals_in_hull) #number of pixels in hull
    
    pixel_vals_in_hull = copy.deepcopy(sci_pixel_vals_in_hull)
    
    if dark_img is not None:
        whered_dark = np.where(hull_mask==1, dark_img, -1000)
        dark_pixel_vals_in_hull = whered_dark[np.where(whered_dark != -1000)]
    else:
        dark_pixel_vals_in_hull = np.zeros(shape=sci_pixel_vals_in_hull.shape)
        
    if bias_img is not None:
        whered_bias = np.where(hull_mask==1, bias_img, -1000)
        bias_pixel_vals_in_hull = whered_bias[np.where(whered_bias != -1000)]
    else:
        bias_pixel_vals_in_hull = np.zeros(shape=sci_pixel_vals_in_hull.shape)
    
    pixel_vals_in_hull = pixel_vals_in_hull - dark_pixel_vals_in_hull - bias_pixel_vals_in_hull
    pixel_vals_in_hull_err = np.sqrt(pixel_vals_in_hull + dark_pixel_vals_in_hull + bias_pixel_vals_in_hull)

    mu = np.mean(pixel_vals_in_hull)
    sigma = np.std(pixel_vals_in_hull)
    
    #print("test", np.sqrt(np.sum(pixel_vals_in_hull)/N))
    
    mu_err = 1/N * np.sqrt(np.sum(pixel_vals_in_hull_err**2))
    #print(pixel_vals_in_hull_err)
    #print(np.sum(pixel_vals_in_hull_err**2))
    #print(np.sqrt(np.sum(pixel_vals_in_hull_err**2)))
    term = np.power((pixel_vals_in_hull-mu)/(sigma*N) * pixel_vals_in_hull_err, 2)
    sigma_err = np.sqrt(np.sum(term))
    
    SNR = np.divide(mu, sigma)
    SNR_err = SNR * np.sqrt((mu_err/mu)**2 + (sigma_err/sigma)**2)
    return mu, mu_err, sigma, sigma_err, SNR, SNR_err

def driver(images_dir, save_boundary_points=False):    
    search_str = images_dir + "/*.tif"
    images_filenames_list = glob.glob(search_str) #list of full filenames of images in images_dir
    
    results_savedir = os.path.join(images_dir, "Results")
    if os.path.isdir(results_savedir) == False: os.mkdir(results_savedir)
    
    min_exp_time_us = 10 #minimum exposure time in [us]
    darks_dir = os.path.join(images_dir, "Darks")
    if os.path.isdir(darks_dir):
        print("Processing darks...")
        dict_of_dark_images = process_darks(darks_dir)
        if min_exp_time_us in dict_of_dark_images.keys():
            print("Bias images found")
        print("Processing darks complete")
    
    #Iterate over all filenames in images_filenames_list
    for filename in images_filenames_list:
        basename = os.path.basename(filename) #file basename, with extension
        basename_no_ext = os.path.splitext(basename)[0] #file basename, without extension
        print("Processing", basename)
        
        #Load the json file which is associated with the image
        json_filename = os.path.splitext(filename)[0] + ".json" #the filename of the associated json file to the image
        json_dict = load_json(json_filename)
        exp_time = json_dict["exposure_time_us"]
        
        img_og = read_12bit_image_saved_as_16bit(filename, verbosity=0) #read the image
        img = subtract_dark_and_bias(img_og, exp_time, dict_of_dark_images, min_exp_time_us)
        
        ###############################################################################
        #APPLY ALPHA SHAPES AND SAVE THE MASK
        #Find the hull mask, which includes all pixels inside the alpha shape boundary
        if save_boundary_points == True:
            hull_mask, boundary_points = hull_mask_from_alpha_shape(img, return_boundary_points=True, verbosity=0) 
        else:
            hull_mask = hull_mask_from_alpha_shape(img, return_boundary_points=False, verbosity=0) 
        #Save the mask; write it to an image
        hull_mask_savename = os.path.join(results_savedir, basename_no_ext+"_mask.bmp")
        cv.imwrite(hull_mask_savename, 255 * hull_mask)
        
        #Apply the mask to the original image
        whered = np.where(hull_mask==1, img, -1000) #-1000 can instead be any arbitrary number which is NOT a valid pixel value
        list_of_pixel_vals_in_hull = whered[np.where(whered != -1000)] #list of pixel values inside the mask
        
        ###############################################################################
        #COMPUTE STATISTICS AND SAVE RESULTS IN NEW JSON FILE
        
        dark_img, bias_img = get_dark_and_bias(exp_time, dict_of_dark_images, min_exp_time_us=min_exp_time_us)
        
        mu, mu_err, sigma, sigma_err, SNR, SNR_err = compute_SNR(hull_mask, img_og, dark_img, bias_img)

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
        
    return True

if __name__ == "__main__":
    #CHANGE THIS LINE AS NEEDED:
    images_dir = os.path.join(os.getcwd(), "Images", "20220526_Square_Test")
    images_dir = os.path.join(os.getcwd(), "Images", "20220531_Stack_Vary")
    images_dir = os.path.join(os.getcwd(), "Images", "20220602_Stack_Vary")
    images_dir = os.path.join(os.getcwd(), "Images", "20220602_Exp_Vary")
    images_dir = os.path.join(os.getcwd(), "Images", "20220603_Tot_Exp_Const")
    #images_dir = os.path.join(os.getcwd(), "Images", "20220607_LDLS_View")
    
    driver(images_dir, save_boundary_points=True)
    
    