#Matthew Leung
#April 2022
#Python 3
"""
Only works in 3 FPS because device specific interface layout is used
"""

from __future__ import print_function

# import standard packages
import sys
import os
import ctypes
import time
from datetime import date
import shutil
from PIL import Image
import numpy as np

# Matrix Vision
from mvIMPACT import acquire
#from mvIMPACT.Common import exampleHelper

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

def read_uint12(data_chunk):
    data = np.frombuffer(data_chunk, dtype=np.uint8)
    fst_uint8, mid_uint8, lst_uint8 = np.reshape(data, (data.shape[0] // 3, 3)).astype(np.uint16).T
    fst_uint12 = (fst_uint8 << 4) + (mid_uint8 >> 4)
    snd_uint12 = ((mid_uint8 % 16) << 8) + lst_uint8
    return np.reshape(np.concatenate((fst_uint12[:, None], snd_uint12[:, None]), axis=1), 2 * fst_uint12.shape[0])



def acquire_frame(filename, exptime=0.0, ampgain=0.0, verbosity=0): 
    import time
    img_list = []
    
    
    #if self._verbose: self._log("Image acqisition start")
    #exptime = min(max(exptime, 0.00001), 20.0)
    devMgr = acquire.DeviceManager()
    printDevices(devMgr)
    pDev = devMgr.getDevice(0)
    
    #In order to use acquire.CameraSettingsBlueCOUGAR, you CANNOT use the
    #GenICam interface! I.e. the following line is not allowed:
    #conditionalSetProperty(pDev.interfaceLayout, acquire.dilGenICam) #Use GenICam
    #Technically, it doesn't really matter if you have this line^
    #What matters is you set to device specific below!
    
    #IMPORTANT: It is required to have one of the two lines below!
    #They are the same thing; conditionalSetProperty does .write on pDev.interfaceLayout
    #Here we set interfaceLayout to device specific instead of GenICam
    conditionalSetProperty(pDev.interfaceLayout, acquire.dilDeviceSpecific)
    pDev.interfaceLayout.write(acquire.dilDeviceSpecific)
    
    conditionalSetProperty(pDev.acquisitionStartStopBehaviour, acquire.assbUser)
    pDev.open()
    fi = acquire.FunctionInterface(pDev)
    result = fi.acquisitionStop()
    
    #pDev.interfaceLayout.getTranslationDictString() #this does nothing but read the a dictionary containing the valid values for this property, for default key of 0
    interfaceLayout_str_value = pDev.interfaceLayout.readS() #the value of the interfaceLayout property, in str format
    interfaceLayout_value = pDev.interfaceLayout.read()
    interfaceLayout_str_value_from_dict = pDev.interfaceLayout.getTranslationDictString(interfaceLayout_value)
    print("The current interface layout is "+interfaceLayout_str_value)
    print("From reading the dictionary, the current interface layout is "+interfaceLayout_str_value_from_dict)
    
    ac = acquire.CameraSettingsBlueCOUGAR(pDev)
    ac.pixelFormat.write(acquire.idpfMono12)
    print(ac.pixelFormat.isWriteable)
    #ac.pixelFormat.write(acquire.idpfMono8)
    print(ac.pixelFormat.readS())
    ac.expose_us.write(800)
    ac.gain_dB.write(0)
    ac.offset_pc.write(0)
    ac.frameRate_Hz.write(60)
    ac.frameDelay_us.write(10)

    gst = time.time() #global start time

    result = fi.acquisitionStart()
    while fi.imageRequestSingle() == acquire.DMR_NO_ERROR:
        if verbosity > 0: print(".",end='')
    if verbosity > 0: print('')
    pPreviousRequest = None
    manuallyStartAcquisitionIfNeeded(pDev, fi)
    for i in range(0,10,1):
        st = time.time()
        requestNr = fi.imageRequestWaitFor(10000) #DO NOT MAKE THIS -1
        print("Requesting image took {} seconds".format(time.time()-st))
        
        if fi.isRequestNrValid(requestNr):
            pRequest = fi.getRequest(requestNr)
            if pRequest.isOK:
                
                st = time.time()
                imgread = pRequest.imageData.read()
                print("pRequest.imageData.read() took {} seconds".format(time.time()-st))
                
                st = time.time()
                cbuf = (ctypes.c_char * pRequest.imageSize.read()).from_address(int(imgread))
                print("cbuf took {} seconds".format(time.time()-st))
                if pRequest.imageChannelBitDepth.read() > 8:
                    channelType = np.uint16
                else:
                    channelType = np.uint8
                    print("WARNING: Camera is acquiring in 8 bit!")
                
                total_size = pRequest.imageHeight.read() * pRequest.imageWidth.read() * pRequest.imageChannelCount.read()
                print(total_size)
                
                st = time.time()
                arr = np.frombuffer(cbuf, dtype=channelType)
                print("np.frombuffer took {} seconds".format(time.time()-st))
                st = time.time()
                arr_rs = arr.reshape(pRequest.imageHeight.read(), pRequest.imageWidth.read(), pRequest.imageChannelCount.read())
                print("reshape took {} seconds".format(time.time()-st))
                
                time_elapsed = time.time() - gst
                frame_rate = (i+1)/time_elapsed
                print("Time elapsed = {} seconds".format(time_elapsed))
                print("FPS = {}".format(frame_rate))
                print("\n")
                
                #print("The number of unique values in the image is {}".format(np.size(np.unique(arr_rs))))
                img_list += [arr_rs.copy()] #must make a copy!
                

                
            if pPreviousRequest != None:
                pPreviousRequest.unlock()
            pPreviousRequest = pRequest
            fi.imageRequestSingle() #must have this
        else:
            print("imageRequestWaitFor failed (" + str(requestNr) + ", " + 
                  acquire.ImpactAcquireException.getErrorCodeAsString(requestNr) + ")")
    result = fi.acquisitionStop()
    pDev.close()
    
    s = np.zeros(shape=(1104,1600))
    for i in range(0,len(img_list),1):
        curr_img = img_list[i]
        print(curr_img.shape)
        s += curr_img[:,:,0]
        
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(s, cmap='viridis')
    plt.colorbar()
    plt.show()
    
    return True 


if __name__ == "__main__":
    acquire_frame(None)





