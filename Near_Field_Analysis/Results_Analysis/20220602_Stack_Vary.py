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
    numeric = [False, False, True, True, True, True, True]
    message_parts = message.split(",")
    
    for i in range(0,len(message_parts),1):
        part = message_parts[i]
        feature_name = feature_names[i]
        is_numeric = numeric[i]
        if is_numeric == True:
            part_splitted = part.split(" ")
            feature_str_rep = part_splitted[1]
            json_dict[feature_name] = re.sub("[^0-9.]", "", feature_str_rep)
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

if __name__ == "__main__":
    images_dir = os.path.join(os.getcwd(), os.pardir, "Images", "20220602_Stack_Vary")
    images_dir_basename = os.path.basename(images_dir)
    results_savedir = os.path.join(images_dir, "Results")
    plots_savedir = os.path.join(results_savedir, "Plots")
    if os.path.isdir(plots_savedir) == False: os.mkdir(plots_savedir)
        
    df = construct_df_from_jsons(results_savedir)

    ###################################################################################
    
    off_df = df[df['MS_ON_OFF'] == "MS OFF"]
    off_num_frames = off_df['number of frames'].to_numpy()
    off_snr = off_df['mean/std'].to_numpy()
    off_unique_num_frames, off_ms_means, off_ms_stds = get_mean_std_points(off_df, feature='number of frames', metric='mean/std')
    
    on_df = df[df['MS_ON_OFF'] == "MS ON"]
    on_num_frames = on_df['number of frames'].to_numpy()
    on_snr = on_df['mean/std'].to_numpy()
    on_unique_num_frames, on_ms_means, on_ms_stds = get_mean_std_points(on_df, feature='number of frames', metric='mean/std')
    
    mpl.rcParams['font.family'] = 'Serif'
    mpl.rcParams['xtick.top'] = True
    mpl.rcParams['ytick.right'] = True
    mpl.rcParams['xtick.minor.visible'] = True
    mpl.rcParams['ytick.minor.visible'] = True
    mpl.rcParams['xtick.direction'] = 'in'
    mpl.rcParams['ytick.direction'] = 'in'
    mpl.rcParams['xtick.major.size'] = 6  # default 3.5
    mpl.rcParams['ytick.major.size'] = 6  # default 3.5

    ###################################################################################

    handles = []
    labels = []

    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    #fig.add_subplot(111, frameon=False) # add a big axis, hide frame
    
    mew = 0.5
    ms = 5
    capsize = 6
    pltcolor = 'dimgrey'
    ax[0].scatter(on_num_frames, on_snr, color='tab:blue', s=25, linewidth=mew, edgecolor='k', label='Scrambler ON', zorder=5)
    ax[0].errorbar(on_unique_num_frames, on_ms_means, yerr=on_ms_stds, fmt='-', markersize=ms, ecolor=pltcolor, elinewidth=1, color=pltcolor, capsize=capsize, markeredgewidth=mew, markeredgecolor='k', zorder=4)
    ax[0].set_ylabel('Signal-to-Noise Ratio')
    ax[0].set_xscale('log')
    #ax[0].set_ylim([7,11])
    curr_handles, curr_labels = ax[0].get_legend_handles_labels()
    handles += curr_handles
    labels += curr_labels
    
    ax[1].set_xlabel('Number of Stacked Frames')
    ax[1].scatter(off_num_frames, off_snr, color='tab:orange', s=25, linewidth=mew, edgecolor='k', label='Scrambler OFF', zorder=5)
    ax[1].errorbar(off_unique_num_frames, off_ms_means, yerr=off_ms_stds, fmt='-', markersize=ms, ecolor=pltcolor, elinewidth=1, color=pltcolor, capsize=capsize, markeredgewidth=mew, markeredgecolor='k', zorder=4, label="Average")
    curr_handles, curr_labels = ax[1].get_legend_handles_labels()
    handles += curr_handles
    labels += curr_labels
    ax[0].legend(handles, labels, loc='lower right', framealpha=1.0)
    
    plt.savefig(os.path.join(plots_savedir,images_dir_basename+'_SNR.png'), bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(plots_savedir,images_dir_basename+'_SNR.pdf'), bbox_inches='tight')
    plt.show()
    
    mpl.rcParams.update(mpl.rcParamsDefault)
    