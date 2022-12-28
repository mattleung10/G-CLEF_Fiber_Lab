#Matthew Leung
#April 2022
#Python 3
"""
With the camera turned on, run this script to continuously acquire images,
and then stack them. This script does not save results. Use this script as a
tool to see if your exposure, gain, etc. settings are correct. Then run
mv_acquire_continuous_save.py to actually acquire and save images.

In this script, I DO NOT use device specific interface. I use GenICam instead.
Use acquire.AcquisitionControl, acquire.AnalogControl,
acquire.ImageFormatControl, instead of acquire.CameraSettingsBlueCOUGAR.

FROM MATRIX VISION SDK WEBSITE:
https://www.matrix-vision.com/manuals/SDK_PYTHON/ChangeDeviceSettings.html
For the GenICam interface layout there will be convenient access objects fro
ALL the feature defined in the GenICam Standard Feature Naming Convention
(SFNC) as well as ALL the custom features defined by MATRIX VISION. So to
access e.g. SFNC compliant features belonging the AcquisitionControl category
this object can be used
"""
from __future__ import print_function

import os
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image
import ctypes
import time
import warnings


# Matrix Vision
from mvIMPACT import acquire

#######################################################################################
#######################################################################################
#Matrix Vision from mvIMPACT.Common.exampleHelper

def printDevices(devMgr, boSilent=False, boAutomaticallyUseGenICamInterface=True):
    msg = ""
    for i in range(devMgr.deviceCount()):
        pDev = devMgr.getDevice(i)
        msg = "[" + str(i) + "]: " + pDev.serial.read() + "(" + pDev.product.read() + ", " + pDev.family.read()
        if pDev.interfaceLayout.isValid:
            if boAutomaticallyUseGenICamInterface == True:
                conditionalSetProperty(pDev.interfaceLayout, acquire.dilGenICam, boSilent)
            msg += ", interface layout: " + pDev.interfaceLayout.readS()
        if pDev.acquisitionStartStopBehaviour.isValid:
            conditionalSetProperty(pDev.acquisitionStartStopBehaviour, acquire.assbUser, boSilent)
            msg += ", acquisition start/stop behaviour: " + pDev.acquisitionStartStopBehaviour.readS()
        if pDev.isInUse:
            msg += ", !!!ALREADY IN USE!!!"
        print(msg + ")")
    
    if msg == "":
        print("No devices were found")
    return True


def supportsValue(prop, value):
    if prop.hasDict:
        validValues = []
        prop.getTranslationDictValues(validValues)
        return value in validValues

    if prop.hasMinValue and prop.getMinValue() > value:
        return False

    if prop.hasMaxValue and prop.getMaxValue() < value:
        return False

    return True

def conditionalSetProperty(prop, value, boSilent=False):
    #Modified to include boolean return
    if prop.isValid and prop.isWriteable and supportsValue(prop, value):
        prop.write(value)
        if boSilent == False:
            print("Property '" + prop.name() + "' set to '" + prop.readS() + "'.")
        return True
    return False

def getNumberFromUser():
    return int(input())

def getDeviceFromUserInput(devMgr, boSilent=False, boAutomaticallyUseGenICamInterface=True):
    for i in range(devMgr.deviceCount()):
        pDev = devMgr.getDevice(i)
        msg = "[" + str(i) + "]: " + pDev.serial.read() + "(" + pDev.product.read() + ", " + pDev.family.read()
        if pDev.interfaceLayout.isValid:
            if boAutomaticallyUseGenICamInterface == True:
                conditionalSetProperty(pDev.interfaceLayout, acquire.dilGenICam, boSilent)
            msg += ", interface layout: " + pDev.interfaceLayout.readS()
        if pDev.acquisitionStartStopBehaviour.isValid:
            conditionalSetProperty(pDev.acquisitionStartStopBehaviour, acquire.assbUser, boSilent)
            msg += ", acquisition start/stop behaviour: " + pDev.acquisitionStartStopBehaviour.readS()
        if pDev.isInUse:
            msg += ", !!!ALREADY IN USE!!!"
        print(msg + ")")

    print("Please enter the number in front of the listed device followed by [ENTER] to open it: ", end='')
    devNr = getNumberFromUser()
    if (devNr < 0) or (devNr >= devMgr.deviceCount()):
        print("Invalid selection!")
        return None
    return devMgr.getDevice(devNr)

def requestENTERFromUser():
    msg = "Press Enter to continue..."
    input(msg)

# Start the acquisition manually if this was requested(this is to prepare the driver for data capture and tell the device to start streaming data)
def manuallyStartAcquisitionIfNeeded(pDev, fi):
    if pDev.acquisitionStartStopBehaviour.read() == acquire.assbUser:
        result = fi.acquisitionStart()
        if result != acquire.DMR_NO_ERROR:
            print("'FunctionInterface.acquisitionStart' returned with an unexpected result: " + str(result) + "(" + acquire.ImpactAcquireException.getErrorCodeAsString(result) + ")")

# Stop the acquisition manually if this was requested
def manuallyStopAcquisitionIfNeeded(pDev, fi):
    if pDev.acquisitionStartStopBehaviour.read() == acquire.assbUser:
        result = fi.acquisitionStop()
        if result != acquire.DMR_NO_ERROR:
            print("'FunctionInterface.acquisitionStop' returned with an unexpected result: " + str(result) + "(" + acquire.ImpactAcquireException.getErrorCodeAsString(result) + ")")


#######################################################################################
#######################################################################################


def acquire_continuous(num_frames, exp_time_us=100, FPS=25, verbosity=0):
    """
    This function continuously acquires images from Matrix Vision camera
    INPUTS:
        ::int:: num_frames   #number of frames to take
        ::float:: exp_time_us #exposure time in microseconds
        ::float:: FPS        #frames per second
    OUTPUT:
        ::list of np.ndarray:: img_list #list of all the individual frames
        ::list of float:: FPS_list      #list of the FPS values for each frame
    """
    img_list = []
    FPS_list = []
    
    if type(num_frames) != int:
        raise ValueError("The number of frames must be an integer")
    if exp_time_us < 10.0:
        raise ValueError("Below minimum exposure time")
    
    devMgr = acquire.DeviceManager()
    print("The available devices are:")
    printDevices(devMgr)
    pDev = devMgr.getDevice(0)
    
    #Set the device interface layout to GenICam
    conditionalSetProperty(pDev.interfaceLayout, acquire.dilGenICam) #Use GenICam
    
    conditionalSetProperty(pDev.acquisitionStartStopBehaviour, acquire.assbUser)
    pDev.open()
    fi = acquire.FunctionInterface(pDev)
    result = fi.acquisitionStop()
    
    interfaceLayout_str_value = pDev.interfaceLayout.readS() #the value of the interfaceLayout property, in str format
    print("The current interface layout is "+interfaceLayout_str_value)
    
    #Acquisition Control
    #IMPORTANT: To use acquire.AcquisitionControl, we MUST use GenICam Interface!
    ac = acquire.AcquisitionControl(pDev)
    #ac.pixelFormat.write(acquire.idpfMono12)
    ac.exposureAuto.writeS('Off') #turn off auto exposure
    ac.exposureMode.writeS("Timed")
    ac.exposureTime.write(exp_time_us) 
    ac.acquisitionFrameRateEnable.writeS('1') #allow FPS to be written
    ac.acquisitionFrameRate.write(FPS)
    
    #Analog Control
    algc = acquire.AnalogControl(pDev)
    algc.gainAuto.writeS('Off') #turn off auto gain
    algc.gain.write(0.0) #0 dB gain
    
    #Image Format Control
    ifc = acquire.ImageFormatControl(pDev)
    ifc.pixelFormat.writeS('Mono12') #12 bit image
    
    gst = time.time() #global start time

    result = fi.acquisitionStart() #start acquisition
    while fi.imageRequestSingle() == acquire.DMR_NO_ERROR:
        if verbosity > 0: print(".",end='')
    if verbosity > 0: print('')
    
    pPreviousRequest = None
    manuallyStartAcquisitionIfNeeded(pDev, fi)
    for i in range(0,num_frames,1):
        if verbosity >= 1: print("Acquiring image {} out of {}...".format(i+1,num_frames))
        
        iteration_st = time.time() #start time for the current iteration
        requestNr = fi.imageRequestWaitFor(10000) #DO NOT MAKE THE ARGUMENT -1; THAT IS BAD PRACTICE
        print("Requesting image took {} seconds".format(time.time()-iteration_st))
        
        if fi.isRequestNrValid(requestNr):
            pRequest = fi.getRequest(requestNr)
            if pRequest.isOK:
                #Read the image data
                st = time.time()
                imgread = pRequest.imageData.read()
                if verbosity >= 2: print("pRequest.imageData.read() took {} seconds".format(time.time()-st))
                
                #Get ctypes buffer
                st = time.time()
                cbuf = (ctypes.c_char * pRequest.imageSize.read()).from_address(int(imgread))
                if verbosity >= 2: print("cbuf took {} seconds".format(time.time()-st))
                if pRequest.imageChannelBitDepth.read() > 8:
                    channelType = np.uint16
                else:
                    channelType = np.uint8
                    warnings.warn("WARNING: Camera is acquiring in 8 bit!")
                
                #Get the total size of the image
                if verbosity >= 2:
                    total_size = pRequest.imageHeight.read() * pRequest.imageWidth.read() * pRequest.imageChannelCount.read()
                    print("Total size of image is {}".format(total_size))
                
                #Convert buffer to np.ndarray
                st = time.time()
                arr = np.frombuffer(cbuf, dtype=channelType)
                if verbosity >= 2: print("np.frombuffer took {} seconds".format(time.time()-st))
                
                #Reshape the np.ndarray to correct dimensions
                st = time.time()
                arr_rs = arr.reshape(pRequest.imageHeight.read(), pRequest.imageWidth.read(), pRequest.imageChannelCount.read())
                if verbosity >= 2: print("reshape took {} seconds".format(time.time()-st))
                if verbosity >= 3:
                    print("The number of unique values in the image is {}".format(np.size(np.unique(arr_rs))))
                    print("Image Statistics:")
                    print("Min:", np.min(arr_rs))
                    print("Max:", np.max(arr_rs))
                    print("Mean:", np.mean(arr_rs))
                    print("StDev:", np.std(arr_rs))
                
                curr_time = time.time()
                
                time_elapsed = curr_time - gst
                frame_rate = (i+1)/time_elapsed
                FPS_list += [frame_rate]
                if verbosity >= 1:
                    curr_iteration_time = curr_time - iteration_st
                    print("This iteration took {} seconds".format(curr_iteration_time))
                    print("Total time elapsed is {} seconds".format(time_elapsed))
                    print("FPS = {}".format(frame_rate))
                    print("\n")
                
                img_list += [arr_rs.copy()] #must make a copy!
                
            if pPreviousRequest != None:
                pPreviousRequest.unlock()
            pPreviousRequest = pRequest
            fi.imageRequestSingle() #must have this
        else:
            print("imageRequestWaitFor failed (" + str(requestNr) + ", " + 
                  acquire.ImpactAcquireException.getErrorCodeAsString(requestNr) + ")")
    
    if verbosity >= 1:
        total_time_elapsed = time.time() - gst
        final_FPS = num_frames / total_time_elapsed
        print("The final FPS is {}".format(final_FPS))
    
    result = fi.acquisitionStop()
    pDev.close()
    
    return img_list, FPS_list

def acquire_continuous_driver(num_frames, exp_time_us, FPS, verbosity=0):

    img_list, FPS_list = acquire_continuous(num_frames, exp_time_us=exp_time_us, FPS=FPS, verbosity=verbosity)
    
    stacked_arr = np.zeros(shape=(1104,1600))
    for i in range(0,len(img_list),1):
        curr_img = img_list[i]
        stacked_arr += curr_img[:,:,0] #add image, get rid of dimension along axis=2
    
    #Normalize the stacked iamge
    stacked_arr_norm = stacked_arr / len(img_list)
    
    #https://stackoverflow.com/questions/63310083/dynamic-range-bit-depth-in-pils-fromarray-function
    arr_astype_16bit = np.array(stacked_arr_norm).astype(np.uint16) #values inside array are still 12 bit de facto, but the array is now 16 bit
    arr16bit = np.multiply(arr_astype_16bit, 2**4)
    
    img16bit = Image.fromarray(arr16bit)
    #img16bit.save('test.tif') #uncomment this to save a test image
    
    plt.figure()
    plt.imshow(stacked_arr, cmap='gray')
    cbar = plt.colorbar()
    cbar.set_label('Value')
    plt.title('Not Normalized Stacked Image')
    plt.show()
    
    return True 

def play_finished_sound(num_beeps=5):
    #Play a sound to indicate that script has finished running
    if os.name == "nt": #if operating system is Windows
        try:
            import winsound
        except ImportError:
            return False
        for i in range(0,num_beeps-1,1):
            winsound.MessageBeep()
            time.sleep(0.3)
        winsound.MessageBeep()
    else:
        return False
    return True

if __name__ == "__main__":
    ###########################################################################
    #CHANGE THESE AS REQUIRED:
    num_frames = 1
    exp_time_us = 60 #minimum is 10us
    FPS = 25
    verbosity=3
    ###########################################################################
    
    acquire_continuous_driver(num_frames, exp_time_us, FPS, verbosity)
    play_finished_sound(num_beeps=3)




