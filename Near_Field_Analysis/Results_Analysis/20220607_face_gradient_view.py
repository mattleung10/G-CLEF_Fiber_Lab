#Matthew Leung
#June 2022

import os
import sys
#Do this to allow for alpha_shape_id_roi.py to be imported:
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import numpy as np
import matplotlib as mpl
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

#######################################################################################
    
if __name__ == "__main__":
    mpl.rcParams['font.family'] = 'Serif'
    
    images_dir = os.path.join(os.getcwd(), "Images", "20220607_LDLS_View")
    results_savedir = os.path.join(images_dir, "Results")
    plots_savedir = os.path.join(results_savedir, "Plots")
    if os.path.isdir(plots_savedir) == False: os.mkdir(plots_savedir)
    
    min_exp_time_us = 10 #minimum exposure time in [us]
    darks_dir = os.path.join(images_dir, "Darks")
    if os.path.isdir(darks_dir):
        print("Processing darks...")
        dict_of_dark_images = process_darks(darks_dir)
        if min_exp_time_us in dict_of_dark_images.keys():
            print("Bias images found")
        print("Processing darks complete")
    
    filenames = [os.path.join(images_dir, "20220607_135110.tif"), #200um Circle
                 os.path.join(images_dir, "20220607_141633.tif"), #square orig
                 os.path.join(images_dir, "20220607_144916.tif"), #600um Circle
                 os.path.join(images_dir, "20220607_160736.tif"), #rect
                 os.path.join(images_dir, "20220607_143357.tif")] #square reverse
    
    scale_factors = [0.2, 0.5, 0.2, 0.6, 0.6]
    
    for i in range(0,len(filenames),1):
        filename = filenames[i]
        img_og = read_12bit_image_saved_as_16bit(filename, verbosity=0) 
        
        basename = os.path.basename(filename) #file basename, with extension
        basename_no_ext = os.path.splitext(basename)[0] #file basename, without extension
        mask_filename = os.path.join(results_savedir, basename_no_ext + '_mask.bmp')
        hull_mask = cv.imread(mask_filename, cv.IMREAD_GRAYSCALE)
        
        #Load the json file which is associated with the image
        json_filename = os.path.splitext(filename)[0] + ".json" #the filename of the associated json file to the image
        json_dict = load_json(json_filename)
        
        #Load the boundary points (for plotting purposes)
        boundary_savename = os.path.join(results_savedir, basename_no_ext+"_boundary.txt")
        boundary_points = np.loadtxt(boundary_savename)
        
        img = subtract_dark_and_bias(img_og, json_dict["exposure_time_us"], dict_of_dark_images, min_exp_time_us)
        
        #Apply the mask to the original image
        whered = np.where(hull_mask==255, img, -1000) #-1000 can instead be any arbitrary number which is NOT a valid pixel value
        list_of_pixel_vals_in_hull = whered[np.where(whered != -1000)] #list of pixel values inside the mask
        face_max = np.max(list_of_pixel_vals_in_hull)
        
        plt.figure(figsize=(8,6))
        cmap1 = copy.copy(plt.cm.coolwarm)
        im = plt.imshow(whered, vmin=face_max*scale_factors[i], cmap=cmap1)
        im.cmap.set_under('white')
        cbar = plt.colorbar(extend="min")
        cbar.ax.set_ylabel("Pixel Value")
        lw = 0.5
        plt.plot(boundary_points[0], boundary_points[1], color='k', linewidth=lw, linestyle='--')
        #Plot line from first point to last point, complete the shape
        #plt.plot([boundary_points[0][0],boundary_points[0][-1]], [boundary_points[1][1],boundary_points[1][-1]], color='k', linewidth=lw)
        
        savename = os.path.join(plots_savedir, basename_no_ext+'_gradient.pdf')
        plt.savefig(savename, bbox_inches='tight', dpi=600)
        savename = os.path.join(plots_savedir, basename_no_ext+'_gradient.png')
        plt.savefig(savename, bbox_inches='tight', dpi=600)
        plt.close('all')
        
        ###############################################################################
        #Plot zoomed in view
        
        plt.figure(figsize=(8,6))
        cmap1 = copy.copy(plt.cm.coolwarm)
        im = plt.imshow(whered, vmin=face_max*scale_factors[i], cmap=cmap1)
        im.cmap.set_under('white')
        cbar = plt.colorbar(extend="min")
        cbar.ax.set_ylabel("Pixel Value")
        lw = 1
        plt.xlim([np.min(boundary_points[0])-25, np.max(boundary_points[0])+25])
        plt.ylim([np.max(boundary_points[1])+25, np.min(boundary_points[1])-25])
        plt.plot(boundary_points[0], boundary_points[1], color='k', linewidth=lw, linestyle='--')
        #Plot line from first point to last point, complete the shape
        #plt.plot([boundary_points[0][0],boundary_points[0][-1]], [boundary_points[1][1],boundary_points[1][-1]], color='k', linewidth=lw)
        
        savename = os.path.join(plots_savedir, basename_no_ext+'_gradient_zoom.pdf')
        plt.savefig(savename, bbox_inches='tight', dpi=600)
        savename = os.path.join(plots_savedir, basename_no_ext+'_gradient_zoom.png')
        plt.savefig(savename, bbox_inches='tight', dpi=600)
        plt.close('all')
        
        ###############################################################################
        H, W = np.shape(img)
        horpoints = np.arange(W)
        vertpoints = np.arange(H)
        hor_px = int((np.min(boundary_points[1])+np.max(boundary_points[1]))/2)
        vert_px = int((np.min(boundary_points[0])+np.max(boundary_points[0]))/2)
        
        plt.figure(figsize=(8,6))
        plt.plot(horpoints, img[hor_px,:])
        plt.ylabel('Pixel Value')
        plt.xlabel('Horizontal Pixel Number')
        plt.title('Slice at $y='+str(hor_px)+'$')
        plt.xlim([np.min(boundary_points[0])-25, np.max(boundary_points[0])+25])
        savename = os.path.join(plots_savedir, basename_no_ext+'_yslice.pdf')
        plt.savefig(savename, bbox_inches='tight', dpi=300)
        savename = os.path.join(plots_savedir, basename_no_ext+'_yslice.png')
        plt.savefig(savename, bbox_inches='tight', dpi=300)
        plt.close('all')
        
        plt.figure(figsize=(8,6))
        plt.plot(vertpoints, img[:,vert_px])
        plt.ylabel('Pixel Value')
        plt.xlabel('Vertical Pixel Number')
        plt.title('Slice at $x='+str(vert_px)+'$')
        plt.xlim([np.min(boundary_points[1])-25, np.max(boundary_points[1])+25])
        savename = os.path.join(plots_savedir, basename_no_ext+'_xslice.pdf')
        plt.savefig(savename, bbox_inches='tight', dpi=300)
        savename = os.path.join(plots_savedir, basename_no_ext+'_xslice.png')
        plt.savefig(savename, bbox_inches='tight', dpi=300)
        plt.close('all')
        
    
    mpl.rcParams.update(mpl.rcParamsDefault)