#https://medium.com/@kshahir2004/streaming-yuv422-yuyv-packed-pixel-format-data-from-gige-vision-camera-with-python-harvesters-ee2e2aaafca0

import numpy as np
import cv2 as cv
from harvesters.core import Buffer, Harvester

# Set width, height and pixel format of frame if you know the details.
WIDTH = 1600  # Image buffer width as per the camera output
HEIGHT = 1104  # Image buffer height as per the camera output
PIXEL_FORMAT = "Mono12p"  # Camera pixel format as per the camera output


h = Harvester()
h.add_file(r"C:\Program Files\MATRIX VISION\mvIMPACT Acquire\bin\x64\mvGenTLProducer.cti") # Add path to mvGenTLProducer.cti
#h.files
h.update()
print(h.device_info_list[0])

io = h.create_image_acquirer(0)
#io.remote_device.node_map.Width.value = WIDTH 
#io.remote_device.node_map.Width.value = WIDTH 
io.remote_device.node_map.PixelFormat.value = PIXEL_FORMAT
#io.remote_device.node_map.AcquisitionFrameRate.value = fps # Set if required 
io.start_acquisition()
#print(len(h.device_info_list))
io.remote_device.node_map.ExposureTime.value = 10
io.remote_device.node_map.ExposureAuto.value = "Off"
#print(dir(io.remote_device.node_map))

i = 0

# Write stream to a file
output_filename = 'video.avi' # Save stream
fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter(output_filename, fourcc, 10.0, (WIDTH, HEIGHT))

#Create loop for streaming
while cv.waitKey(1) != 27:

    Buffer = io.fetch_buffer(timeout=-1)    
    component = Buffer.payload.components[0]
    #print(component.width)
    if component.width == WIDTH: # To make sure the correct size frames are passed for converting
        original = component.data.reshape(HEIGHT, WIDTH, 1)
        img = original.copy() # To prevent isues due to buffer queue
        #print(size(img))
        image = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
        #image_scaled = cv.normalize(img, dst=None, alpha=0, beta=65535, norm_type=cv.NORM_MINMAX)
        image_scaled = (img / 2**12 * 255).astype('uint8')
        image = image_scaled
        # Place your trained model here with required script when running computer vision model on this stream.      
               
        cv.imshow('img', image)
        out.write(cv.resize(image, (WIDTH, HEIGHT)))
        Buffer.queue()
        #time.sleep(0.03)
        i +=1
    else:
        i +=1


out.release()    
io.stop_acquisition()
io.destroy()
h.reset()
cv.destroyAllWindows()