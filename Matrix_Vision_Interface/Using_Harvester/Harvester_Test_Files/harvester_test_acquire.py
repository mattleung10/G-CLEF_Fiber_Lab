#Matthew Leung
#Code last updated: March 11, 2022
"""
Script for testing purposes; acquire images using Harvester Python library
"""

import numpy as np
import cv2 as cv
import warnings
from harvesters.core import Harvester

if __name__ == "__main__":
    WIDTH = 1600  #image buffer width
    HEIGHT = 1104  #image buffer height
    
    cti_filename = r"C:\Program Files\MATRIX VISION\mvIMPACT Acquire\bin\x64\mvGenTLProducer.cti"
    h = Harvester()
    h.add_file(cti_filename) # Add path to mvGenTLProducer.cti
    h.update()
    
    print("There are {} devices connected, which are:".format(len(h.device_info_list)))
    for i in range(0,len(h.device_info_list),1):
        print("Device {}: {}".format(i,h.device_info_list[i]))
    
    list_index = 0
    print("Creating image acquirer instance for Device {}...".format(list_index))
    ia = h.create_image_acquirer(list_index=list_index)
    
    ia.remote_device.node_map.PixelFormat.value = 'Mono12p' #MUST BE BEFORE START ACQUISITION
    ia.remote_device.node_map.AcquisitionFrameRateEnable.value = True
    ia.remote_device.node_map.AcquisitionFrameRate.value = 10
    ia.start_acquisition()
    ia.remote_device.node_map.ExposureAuto.value = "Off"
    ia.remote_device.node_map.ExposureTime.value = 100
    
    buffer = ia.fetch_buffer()
    buffer.queue()

    #Create loop for streaming
    while cv.waitKey(1) != 27:

        buffer = ia.fetch_buffer()    
        component = buffer.payload.components[0]
        
        #print('ExposureMode', ia.remote_device.node_map.ExposureMode.value)
        #print('ExposureTimeSelector', ia.remote_device.node_map.ExposureTimeSelector.value)
        #print('TriggerSelector', ia.remote_device.node_map.TriggerSelector.value)
        #print('TriggerMode', ia.remote_device.node_map.TriggerMode.value)
        #print('TriggerActivation', ia.remote_device.node_map.TriggerActivation.value)
        #print('TriggerDelay', ia.remote_device.node_map.TriggerDelay.value)
        
        #if component.width == WIDTH: # To make sure the correct size frames are passed for converting
        if True:
            original = component.data.reshape(HEIGHT, WIDTH, int(component.num_components_per_pixel))
            img = original.copy() # To prevent isues due to buffer queue
            #print(size(img))
            #image = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
            
            curr_PixelFormat = ia.remote_device.node_map.PixelFormat.value
            
            if curr_PixelFormat == 'Mono8' or curr_PixelFormat == 'Mono8p':
                warnings.warn('WARNING: Camera is taking images in 8 bit!')
                image = img
            elif curr_PixelFormat == 'Mono12' or curr_PixelFormat == 'Mono12p':
                image_scaled = (img / (2**12-1) * 255).astype('uint8')
                image = image_scaled
            else:
                warnings.warn('WARNING: Camera is taking images in neither 8 nor 12 bit!')
                image = img
            
            #image = img
            # Place your trained model here with required script when running computer vision model on this stream.      
            
            #image = cv.resize(image, (int(image.shape[1]/2), int(image.shape[0]/2)))
            cv.imshow('img', image)
            buffer.queue()
    
    
    ia.stop_acquisition()
    ia.destroy()
    h.reset()
