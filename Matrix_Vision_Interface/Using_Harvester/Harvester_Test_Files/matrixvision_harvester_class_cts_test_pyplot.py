#Matthew Leung
#Code last updated: March 14, 2022
"""
Script for testing purposes; continuously acquire image using Harvester Python
library, and display using matplotlib interactive mode.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
#import cv2 as cv
from matrixvision_harvester_class import matrixvision_harvester_class as matrixvision_harvester_class

if __name__ == "__main__":
    cti_filename = r"C:\Program Files\MATRIX VISION\mvIMPACT Acquire\bin\x64\mvGenTLProducer.cti"
    
    camera_obj = matrixvision_harvester_class()
    camera_obj.init_camera(cti_filename)
    camera_obj.start_camera_acquisition(pixel_format='Mono8')
    camera_obj.set_min_exposure_time(10)
    camera_obj.set_exposure(exp_time=10)
    camera_obj.set_frame_rate(3)
    
    img = camera_obj.get_snapshot_np_array()
    camera_obj.buffer.queue()
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8,4), constrained_layout=True)
    
    #Plot the original image
    im0 = ax[0].imshow(img)
    ax[0].set_title('Original Image')
    
    #Plot the cropped region
    im1 = ax[1].imshow(img)
    ax[1].set_title('Fitted Ellipse')

    plt.ion()
    try:
        while True:
            img = camera_obj.get_snapshot_np_array()
            camera_obj.buffer.queue()
            del camera_obj.buffer
            
            #Update original image
            im0.set_data(img)
            im1.set_data(img)
            plt.pause(0.05)
    except KeyboardInterrupt:
        pass

    plt.ioff()
    plt.show()
            
    camera_obj.done_camera()
    print("Done!")
