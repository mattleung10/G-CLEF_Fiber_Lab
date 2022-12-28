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
import pandas as pd
import glob
import json
import copy
import re

def construct_df_from_jsons(results_savedir):
    """
    Constructs a pd.DataFrame from the JSON files containing the statistics
    of each image. An assumption is that results_savedir only contains the
    desired JSON files. Calls the process_message function, which processes the
    "message" field of the JSON file.
    INPUT:
        ::str:: results_savedir
    OUTPUT:
        ::pd.DataFrame:: df
    """
    search_str = results_savedir + "/*.json"
    json_filenames_list = glob.glob(search_str) #list of full filenames of images in images_dir
    
    list_of_dict = []
    
    #Iterate over all filenames in json_filenames_list
    for json_savename in json_filenames_list:
        with open(json_savename, 'r') as f: #open the json file and load into a dict
            json_dict = json.load(f) #this is a dict
        
        processed_dict = process_message(json_dict)
        list_of_dict += [processed_dict]
        
    df = pd.DataFrame(list_of_dict)
    return df

def process_message(json_dict_og):
    """
    Processes the "message" field of the JSON file.
    This function will have to be modified based on your message convention.
    INPUT:
        ::dict:: json_dict_og
    OUTPUT:
        ::dict:: json_dict
    """
    json_dict = copy.deepcopy(json_dict_og)
    message = json_dict["message"]
    feature_names = ["source", "MS_ON_OFF", "shaft_dist", "amplitude", "period", "delay", "ND"]
    numeric = [False, False, False, True, True, True, True]
    message_parts = message.split(",")
    
    for i in range(0,len(message_parts),1):
        part = message_parts[i]
        feature_name = feature_names[i]
        is_numeric = numeric[i]
        if is_numeric == True:
            part_splitted = part.split(" ")
            feature_str_rep = part_splitted[1]
            json_dict[feature_name] = float(re.sub("[^0-9.]", "", feature_str_rep))
        else:
            if part[0] == " ": part = part[1:]
            json_dict[feature_name] = part
            
    return json_dict

#######################################################################################

def get_mean_std_points(df, feature, metric):
    """
    For each unique entry in <feature>, finds the mean and standard devation
    of <metric>
    INPUTS:
        ::pd.DataFrame:: df
        ::str:: feature
        ::str:: metric
    """
    #feature_vals is a np.ndarray containing all unique entries in the column <feature>
    feature_vals = df[feature].unique()
    feature_vals_list = feature_vals.tolist()
    
    means_list = []
    stds_list = []
    for val in feature_vals_list:
        #df_val is a pd.DataFrame containing rows of df where the column <feature> has value <val>
        df_val = df[df[feature] == val]
        metric_vals = df_val[metric].to_numpy()
        
        means_list += [np.mean(metric_vals)]
        stds_list += [np.std(metric_vals)]
    
    return feature_vals, np.array(means_list), np.array(stds_list)


#######################################################################################

def exp_to_frames(exp_time):
    return 500 / exp_time

def frames_to_exp(num_frames):
    return 500 / num_frames

if __name__ == "__main__":
    images_dir = os.path.join(os.getcwd(), os.pardir, "Images", "20220617_Twist")
    images_dir_basename = os.path.basename(images_dir)
    results_savedir = os.path.join(images_dir, "Results")
    plots_savedir = os.path.join(results_savedir, "Plots")
    if os.path.isdir(plots_savedir) == False: os.mkdir(plots_savedir)
        
    df = construct_df_from_jsons(results_savedir)

    ###################################################################################
    
    off_df = df[df['MS_ON_OFF'] == "MS OFF"]
    off_snr = off_df['mean/std'].to_numpy()

    on_df = df[df['MS_ON_OFF'] == "MS ON"]
    on_snr = on_df['mean/std'].to_numpy()

    mpl.rcParams['font.family'] = 'Serif'
    #mpl.rcParams['xtick.top'] = True
    mpl.rcParams['ytick.right'] = True
    mpl.rcParams['xtick.minor.visible'] = True
    mpl.rcParams['ytick.minor.visible'] = True
    mpl.rcParams['xtick.direction'] = 'in'
    mpl.rcParams['ytick.direction'] = 'in'
    mpl.rcParams['xtick.major.size'] = 6  # default 3.5
    mpl.rcParams['ytick.major.size'] = 6  # default 3.5

    ###################################################################################

    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].boxplot(off_snr)
    ax[0].set_xticklabels(['Scrambler OFF'])
    ax[0].set_ylabel('Signal-to-Noise Ratio')
    ax[1].boxplot(on_snr)
    ax[1].set_xticklabels(['Scrambler ON'])
    fig.tight_layout() 
    
    plt.savefig(os.path.join(plots_savedir,images_dir_basename+'_SNR.png'), bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(plots_savedir,images_dir_basename+'_SNR.pdf'), bbox_inches='tight')
    plt.show()
    
    mpl.rcParams.update(mpl.rcParamsDefault)
    
    