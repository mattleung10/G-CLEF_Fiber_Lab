#Matthew Leung
#Code last updated: March 22, 2022

import numpy as np
import matplotlib.pyplot as plt
from matrixvision_harvester_class import matrixvision_harvester_class as matrixvision_harvester_class

if __name__ == "__main__":
    cti_filename = r"C:\Program Files\MATRIX VISION\mvIMPACT Acquire\bin\x64\mvGenTLProducer.cti"

    camera_obj = matrixvision_harvester_class()
    camera_obj.init_camera(cti_filename)
    camera_obj.start_camera_acquisition(pixel_format='Mono12')
    camera_obj.set_min_exposure_time(10)
    camera_obj.set_exposure(exp_time=10)
    camera_obj.set_frame_rate(5) #Set to 10 FPS
    
    img = camera_obj.get_snapshot_np_array()
    image = img.copy()
    
    print(image)
    print("Maximum pixel value in image is {}".format(np.max(image)))
    print("Minimum pixel value in image is {}".format(np.min(image)))
    print("Number of unique values is {}".format(np.unique(image).size))
    
    plt.figure()
    plt.imshow(image, cmap='gray')
    plt.colorbar()
    plt.show()
    
    plt.figure()
    plt.hist(image.ravel(), 2**12)
    plt.title('Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.show()
    
    camera_obj.buffer.queue()
    del camera_obj.buffer
    