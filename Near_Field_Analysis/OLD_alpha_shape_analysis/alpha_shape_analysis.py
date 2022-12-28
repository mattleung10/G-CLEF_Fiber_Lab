#Matthew Leung
#May 2022
"""
This script contains driver code for functions from alpha_shape_id_roi.py.
Use this script on a directory of images to find the alpha shape of the fiber
face in each image.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import glob
import json
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

def driver(images_dir):
    search_str = images_dir + "/*.tif"
    images_filenames_list = glob.glob(search_str) #list of full filenames of images in images_dir
    
    results_savedir = os.path.join(images_dir, "Results")
    if os.path.isdir(results_savedir) == False: os.mkdir(results_savedir)
    
    #Iterate over all filenames in images_filenames_list
    for filename in images_filenames_list:
        basename = os.path.basename(filename) #file basename, with extension
        basename_no_ext = os.path.splitext(basename)[0] #file basename, without extension
        print("Processing", basename)
        
        #Load the json file which is associated with the image
        json_filename = os.path.splitext(filename)[0] + ".json" #the filename of the associated json file to the image
        json_dict = load_json(json_filename)
        
        img = read_12bit_image_saved_as_16bit(filename, verbosity=0) #read the image
        
        ###############################################################################
        #APPLY ALPHA SHAPES AND SAVE THE MASK
        #Find the hull mask, which includes all pixels inside the alpha shape boundary
        hull_mask = hull_mask_from_alpha_shape(img, verbosity=0) 
        #Save the mask; write it to an image
        hull_mask_savename = os.path.join(results_savedir, basename_no_ext+"_mask.bmp")
        cv.imwrite(hull_mask_savename, 255 * hull_mask)
        
        #Apply the mask to the original image
        whered = np.where(hull_mask==1, img, -1000) #-1000 can instead be any arbitrary number which is NOT a valid pixel value
        list_of_pixel_vals_in_hull = whered[np.where(whered != -1000)] #list of pixel values inside the mask
        
        ###############################################################################
        #COMPUTE STATISTICS AND SAVE RESULTS IN NEW JSON FILE
        mean = np.mean(list_of_pixel_vals_in_hull)
        std = np.std(list_of_pixel_vals_in_hull)

        #Save results in dict; add onto a copy of the original json file
        json_dict['pixels_in_hull'] = len(list_of_pixel_vals_in_hull)
        json_dict['mean'] = mean
        json_dict['std'] = std
        json_dict['mean/std'] = mean/std

        #Save json file in results directory        
        json_savename = os.path.join(results_savedir, basename_no_ext+"_stats.json")
        save_json(json_savename, json_dict)
        
    return True

if __name__ == "__main__":
    #CHANGE THIS LINE AS NEEDED:
    images_dir = os.path.join(os.getcwd(), "Images", "20220526_Square_Test")
    images_dir = os.path.join(os.getcwd(), "Images", "20220531_Stack_Vary")
    
    driver(images_dir)
    
    