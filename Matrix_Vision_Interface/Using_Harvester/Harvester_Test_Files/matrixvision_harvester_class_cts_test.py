#Matthew Leung
#Code last updated: March 14, 2022
"""
Script for testing purposes; continuously acquire image using Harvester Python
library, and display using matplotlib interactive mode OR OpenCV imshow.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2 as cv
from matrixvision_harvester_class import matrixvision_harvester_class as matrixvision_harvester_class

if __name__ == "__main__":
    cti_filename = r"C:\Program Files\MATRIX VISION\mvIMPACT Acquire\bin\x64\mvGenTLProducer.cti"
    
    camera_obj = matrixvision_harvester_class()
    camera_obj.init_camera(cti_filename)
    camera_obj.start_camera_acquisition(pixel_format='Mono8')
    camera_obj.set_min_exposure_time(10)
    camera_obj.set_exposure(exp_time=10)
    camera_obj.set_frame_rate(5)
    
    method = 'cv'
    method = 'pp'
    
    if method == 'pp': #use matplotlib.pyplot interactive mode to display image
        img = camera_obj.get_snapshot_np_array()
        camera_obj.buffer.queue()
        plt.figure()
        im = plt.gca().imshow(img)
    
        plt.ion()
        try:
            while True:
                img = camera_obj.get_snapshot_np_array()
                camera_obj.buffer.queue()
                del camera_obj.buffer
                
                #Update original image
                im.set_data(img)
                plt.pause(0.05)
        except KeyboardInterrupt:
            pass
    
        plt.ioff()
        plt.show()
    else: #use OpenCV imshow to display image
        while cv.waitKey(1) != 27:
            img = camera_obj.get_snapshot_np_array()
            camera_obj.buffer.queue()
            
            #If needed, scale the plot image
            if camera_obj.bit_depth == 12:
                plot_image = (img / (2**12-1) * 255).astype('uint8') #convert to 8 bit image
            elif camera_obj.bit_depth == 8:
                plot_image = img
            
            cv.imshow('Image', plot_image)
            
    camera_obj.done_camera()
    print("Done!")
