#Matthew Leung
#June 2022

import os
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image
import copy

def read_12bit_image_saved_as_16bit(filename, verbosity=0):
    """
    Reads a 12 bit image which was saved as a 16 bit image
    """
    img16bit = Image.open(filename)
    arr16bit = np.asarray(img16bit)
    arr12bit = np.divide(arr16bit, 2**4)
    if verbosity >= 1:
        num_unique_elems = np.size(np.unique(arr12bit))
        print("Image has {} unique elements".format(num_unique_elems))
        print("Image maximum value is {}".format(np.max(arr12bit)))
        print("Image minimum value is {}".format(np.min(arr12bit)))
    return arr12bit

if __name__ == "__main__":
    images_dir = os.path.join(os.getcwd(), os.pardir, "Images", "20220609_Flats")
    images_dir_basename = os.path.basename(images_dir)
    plots_savedir = os.path.join(images_dir, "Results")
    if os.path.isdir(plots_savedir) == False: os.mkdir(plots_savedir)
    
    filename = os.path.join(images_dir, "20220609_110443.tif")
    
    basename = os.path.basename(filename) #file basename, with extension
    basename_no_ext = os.path.splitext(basename)[0] #file basename, without extension
        
    img = read_12bit_image_saved_as_16bit(filename)
    
    plt.figure(figsize=(8,6))
    cmap1 = copy.copy(plt.cm.coolwarm)
    im = plt.imshow(img, vmin=np.mean(img)-3*np.std(img), cmap=cmap1)
    im.cmap.set_under('black')
    cbar = plt.colorbar(extend="min")
    cbar.ax.set_ylabel("Pixel Value")
    
    savename = os.path.join(plots_savedir, basename_no_ext+'_plot.pdf')
    plt.savefig(savename, bbox_inches='tight', dpi=600)
    savename = os.path.join(plots_savedir, basename_no_ext+'_plot.png')
    plt.savefig(savename, bbox_inches='tight', dpi=600)
    plt.close('all')
    