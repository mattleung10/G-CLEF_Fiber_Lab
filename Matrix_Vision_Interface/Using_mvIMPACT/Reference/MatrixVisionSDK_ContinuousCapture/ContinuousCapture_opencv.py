from __future__ import print_function
import os
import platform
import sys

import cv2 as cv
import numpy as np
import ctypes
import matplotlib.pyplot as plt

# import all the stuff from mvIMPACT Acquire into the current scope
from mvIMPACT import acquire
# import all the mvIMPACT Acquire related helper function such as 'conditionalSetProperty' into the current scope
# If you want to use this module in your code feel free to do so but make sure the 'Common' folder resides in a sub-folder of your project then
from mvIMPACT.Common import exampleHelper
 
# For systems with NO mvDisplay library support
#import ctypes
#import Image
#import numpy

def read_uint12(data_chunk):
    #https://stackoverflow.com/questions/44735756/python-reading-12-bit-binary-files/51967333#51967333
    data = np.frombuffer(data_chunk, dtype=np.uint8)
    fst_uint8, mid_uint8, lst_uint8 = np.reshape(data, (data.shape[0] // 3, 3)).astype(np.uint16).T
    fst_uint12 = (fst_uint8 << 4) + (mid_uint8 >> 4)
    snd_uint12 = ((mid_uint8 % 16) << 8) + lst_uint8
    return np.reshape(np.concatenate((fst_uint12[:, None], snd_uint12[:, None]), axis=1), 2 * fst_uint12.shape[0])


    
#####################################################################################################

devMgr = acquire.DeviceManager()
pDev = exampleHelper.getDeviceFromUserInput(devMgr)
if pDev == None:
    exampleHelper.requestENTERFromUser()
    sys.exit(-1)
pDev.open()
 
print("Please enter the number of buffers to capture followed by [ENTER]: ", end='')
framesToCapture = exampleHelper.getNumberFromUser()
if framesToCapture < 1:
    print("Invalid input! Please capture at least one image")
    sys.exit(-1)
 
# The mvDisplay library is only available on Windows systems for now
isDisplayModuleAvailable = platform.system() == "Windows"
if isDisplayModuleAvailable:
    display = acquire.ImageDisplayWindow("A window created from Python")
else:
    print("The mvIMPACT Acquire display library is not available on this('" + platform.system() + "') system. Consider using the PIL(Python Image Library) and numpy(Numerical Python) packages instead. Have a look at the source code of this application to get an idea how.")
 
fi = acquire.FunctionInterface(pDev)
statistics = acquire.Statistics(pDev)
 
while fi.imageRequestSingle() == acquire.DMR_NO_ERROR:
    print("Buffer queued")
pPreviousRequest = None


################################


emptyimg = np.zeros(shape=(1104, 1600, 1))
plt.figure()
ax = plt.gca()
im = ax.imshow(emptyimg)

plt.ion()

################################

 
exampleHelper.manuallyStartAcquisitionIfNeeded(pDev, fi)
for i in range(framesToCapture):
    requestNr = fi.imageRequestWaitFor(10000)
    if fi.isRequestNrValid(requestNr):
        pRequest = fi.getRequest(requestNr)
        if pRequest.isOK:
            if i%100 == 0:
                print("Info from " + pDev.serial.read() +
                         ": " + statistics.framesPerSecond.name() + ": " + statistics.framesPerSecond.readS() +
                         ", " + statistics.errorCount.name() + ": " + statistics.errorCount.readS() +
                         ", " + statistics.captureTime_s.name() + ": " + statistics.captureTime_s.readS())
            #if isDisplayModuleAvailable:
            #    display.GetImageDisplay().SetImage(pRequest)
            #    display.GetImageDisplay().Update()
            # For systems with NO mvDisplay library support
            cbuf = (ctypes.c_char * pRequest.imageSize.read()).from_address(int(pRequest.imageData.read()))
            channelType = np.uint16 if pRequest.imageChannelBitDepth.read() > 8 else np.uint8
            #arr = np.fromstring(cbuf, dtype = channelType)
            #arr = np.frombuffer(cbuf, channelType)  
            arr = read_uint12(cbuf) #NEED THIS FOR 12 BIT BUFFER!

            #arr.shape = (pRequest.imageHeight.read(), pRequest.imageWidth.read(), pRequest.imageChannelCount.read())
            arr_rs = arr.reshape(pRequest.imageHeight.read(), pRequest.imageWidth.read(), pRequest.imageChannelCount.read())

            img = arr_rs
            #img = Image.fromarray(arr)
            
            #im.set_data(img)
            
            #im.set_clim(np.min(img), np.max(img))
            #cv.imshow('window', img)
            
            plt.figure()
            plt.imshow(img)
            plt.show()
            
                
        if pPreviousRequest != None:
            pPreviousRequest.unlock()
        pPreviousRequest = pRequest
        #fi.imageRequestSingle()
    else:
        # Please note that slow systems or interface technologies in combination with high resolution sensors
        # might need more time to transmit an image than the timeout value which has been passed to imageRequestWaitFor().
        # If this is the case simply wait multiple times OR increase the timeout(not recommended as usually not necessary
        # and potentially makes the capture thread less responsive) and rebuild this application.
        # Once the device is configured for triggered image acquisition and the timeout elapsed before
        # the device has been triggered this might happen as well.
        # The return code would be -2119(DEV_WAIT_FOR_REQUEST_FAILED) in that case, the documentation will provide
        # additional information under TDMR_ERROR in the interface reference.
        # If waiting with an infinite timeout(-1) it will be necessary to call 'imageRequestReset' from another thread
        # to force 'imageRequestWaitFor' to return when no data is coming from the device/can be captured.
        print("imageRequestWaitFor failed (" + str(requestNr) + ", " + acquire.ImpactAcquireException.getErrorCodeAsString(requestNr) + ")")
exampleHelper.manuallyStopAcquisitionIfNeeded(pDev, fi)
exampleHelper.requestENTERFromUser()

plt.ioff()
plt.show()