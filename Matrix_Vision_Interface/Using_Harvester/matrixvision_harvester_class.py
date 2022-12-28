#Matthew Leung
#Code last modified: March 18, 2022
"""
Class to control Matrix Vision camera (or any other GenICam compliant camera)
using the Python Harvester library: https://github.com/genicam/harvesters
"""

import numpy as np
import cv2 as cv
import time
import warnings
from harvesters.core import Harvester

class matrixvision_harvester_class:
    def __init__(self):
        self.h = None #instance of the harvesters.core.Harvester class
        self.ia = None #image acquirer object
        self.buffer = None #image buffer object
        self.WIDTH = None #image width
        self.HEIGHT = None #image height
        self.bit_depth = None #integer; number of bits is 2^(bit_depth)
        self.min_exp_time = None #minimum exposure time
    
        return None
    
    def init_camera(self, cti_filename):
        #Initialize the camera
        self.h = Harvester()
        self.h.add_file(cti_filename) # Add path to mvGenTLProducer.cti
        self.h.update()
        
        print("There are {} devices connected, which are:".format(len(self.h.device_info_list)))
        for i in range(0,len(self.h.device_info_list),1):
            print("Device {}: {}".format(i,self.h.device_info_list[i]))
        return True
    
    def start_camera_acquisition(self, pixel_format='Mono12p', list_index=0):
        #Start the camera acquisition
        
        print("Creating image acquirer instance for Device {}...".format(list_index))
        self.ia = self.h.create_image_acquirer(list_index=list_index)
        self.ia.remote_device.node_map.PixelFormat.value = pixel_format #MUST BE BEFORE START ACQUISITION
        #self.ia.remote_device.node_map.ChunkPixelFormat.value = pixel_format
        
        self.ia.start_acquisition()
        self.ia.remote_device.node_map.ExposureAuto.value = "Off"
        self.WIDTH = self.ia.remote_device.node_map.SensorWidth.value
        self.HEIGHT = self.ia.remote_device.node_map.SensorHeight.value
        return True
    
    def done_camera(self):
        #Finished using the camera; clean up
        self.ia.stop_acquisition()
        self.ia.destroy()
        self.h.reset()
        return True
    
    def set_min_exposure_time(self, min_exp_time):
        #Set minimum exposure time
        self.min_exp_time = min_exp_time
        return True
    
    def set_exposure(self, exp_time):
        #Set the camera exposure time
        if exp_time < self.min_exp_time:
            warnings.warn("WARNING: Exposure time less than minimum exposure time allowed")
            return False
        self.ia.remote_device.node_map.ExposureAuto.value = "Off"
        self.ia.remote_device.node_map.ExposureTime.value = exp_time
        return True
    
    def set_frame_rate(self, frame_rate):
        #Set the camera frame rate
        self.ia.remote_device.node_map.AcquisitionFrameRateEnable.value = True
        self.ia.remote_device.node_map.AcquisitionFrameRate.value = frame_rate
        return True
    
    def get_snapshot_np_array(self):
        #Get a single frame; returns a np.ndarray
        
        self.buffer = self.ia.fetch_buffer()    
        component = self.buffer.payload.components[0]
        if component.width == self.WIDTH: #To make sure the correct size frames are passed for converting
            original = component.data.reshape(self.HEIGHT, self.WIDTH, int(component.num_components_per_pixel))
            img = original.copy() #To prevent isues due to buffer queue
            
            curr_PixelFormat = self.ia.remote_device.node_map.PixelFormat.value
            
            if curr_PixelFormat == 'Mono8' or curr_PixelFormat == 'Mono8p':
                warnings.warn('WARNING: Camera is taking images in 8 bit!')
                self.bit_depth = 8
            elif curr_PixelFormat == 'Mono12' or curr_PixelFormat == 'Mono12p':
                self.bit_depth = 12
            else:
                warnings.warn('WARNING: Camera is taking images in neither 8 nor 12 bit!')
                self.bit_depth = 0
            
            return img
        else:
            return np.zeros(shape=(self.HEIGHT, self.WIDTH, int(component.num_components_per_pixel)))
        
        
if __name__ == "__main__":
    cti_filename = r"C:\Program Files\MATRIX VISION\mvIMPACT Acquire\bin\x64\mvGenTLProducer.cti"
    
    camera_obj = matrixvision_harvester_class()
    camera_obj.init_camera(cti_filename)
    camera_obj.start_camera_acquisition()
    camera_obj.set_min_exposure_time(10)
    camera_obj.set_exposure(exp_time=10)
    img = camera_obj.get_snapshot_np_array()
    camera_obj.buffer.queue()
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.colorbar()
    plt.show()
    camera_obj.done_camera()
    