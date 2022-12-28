#Matthew Leung
#Code last updated: April 13, 2022
"""
This script is a test.
It opens a 16 bit image in tif format, which was saved by 
mv_acquire_test_cts_ac.py. The image is actually 12 bit, but just saved in
16 bit format by PIL.
"""

import numpy as np
import PIL.Image as Image

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
    filename = "test.tif"
    read_12bit_image_saved_as_16bit(filename, verbosity=1)

