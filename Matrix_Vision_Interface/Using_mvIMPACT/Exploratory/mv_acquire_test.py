#Matthew Leung
#March 2022
#Python 3

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
    
    #ac = acquire.AcquisitionControl(pDev)
    #ac.pixelFormat.write(acquire.idpfMono12)
    
    #pDev.interfaceLayout.getTranslationDictString() #this does nothing but read the a dictionary containing the valid values for this property, for default key of 0
    interfaceLayout_str_value = pDev.interfaceLayout.readS() #the value of the interfaceLayout property, in str format
    interfaceLayout_value = pDev.interfaceLayout.read()
    interfaceLayout_str_value_from_dict = pDev.interfaceLayout.getTranslationDictString(interfaceLayout_value)
    print("The current interface layout is "+interfaceLayout_str_value)
    print("From reading the dictionary, the current interface layout is "+interfaceLayout_str_value_from_dict)
    
    ac = acquire.CameraSettingsBlueCOUGAR(pDev)
    ac.pixelFormat.write(acquire.idpfMono12)
    ac.expose_us.write(800)
    ac.gain_dB.write(0)
    ac.offset_pc.write(0)
    #ifc = acquire.ImageFormatControl(pDev , "Base")
    #ifc.pixelFormat.write("Mono12")

    
    #csbd = acquire.CameraSettingsBlueDevice(pDev)
    #csbd = acquire.CameraSettingsBlueCOUGAR(pDev)
    #csbd.expose_us.write(int(exptime * 1000000))
    #csbd.autoControlMode.write(0)
    #csbd.autoGainControl.write(0)
    #csbd.gain_dB.write(float(ampgain)) #Electronic gain in dB
    #csbd.pixelFormat.write(acquire.idpfMono12)
    #csbd.offset_pc.write(float(offset_pc)) #A float property defining the analogue sensor offset in percent of the allowed range
    #csbd.offset_pc.write(float(10))
    #conditionalSetProperty(pDev.acquisitionStartStopBehaviour, acquire.assbUser)
    result = fi.acquisitionStart()
    while fi.imageRequestSingle() == acquire.DMR_NO_ERROR:
        if verbosity > 0: print(".",end='')
    if verbosity > 0: print('')
    pPreviousRequest = None
    manuallyStartAcquisitionIfNeeded(pDev, fi)
    requestNr = fi.imageRequestWaitFor(-1)
    if fi.isRequestNrValid(requestNr):
        pRequest = fi.getRequest(requestNr)
        if pRequest.isOK:
            pRequest.imageData.read()
            
            #cbuf = (ctypes.c_char * pRequest.imageSize.read()).from_address(int(pRequest.imageData.read()))
            #channelType = numpy.uint16
            #arr = numpy.frombuffer(cbuf, dtype = numpy.uint16)
            #arr.shape = (pRequest.imageHeight.read(), pRequest.imageWidth.read(), pRequest.imageChannelCount.read())
            #pixel_data = Image.fromarray(arr[:,:,0])
            #print(pixel_data)
            
            cbuf = (ctypes.c_char * pRequest.imageSize.read()).from_address(int(pRequest.imageData.read()))
            if pRequest.imageChannelBitDepth.read() > 8:
                channelType = np.uint16
            else:
                channelType = np.uint8
                print("WARNING: Camera is acquiring in 8 bit!")
            
            total_size = pRequest.imageHeight.read() * pRequest.imageWidth.read() * pRequest.imageChannelCount.read()
            print(total_size)
            
            arr = np.frombuffer(cbuf, dtype=channelType) #Only works for not 12 bit <-- Not true?
            #if arr.size < total_size:
            #    arr = read_uint12(cbuf) #NEED THIS FOR 12 BIT BUFFER!
            #else:
            #    print("WARNING: Camera is acquiring in 8 bit!")
            arr_rs = arr.reshape(pRequest.imageHeight.read(), pRequest.imageWidth.read(), pRequest.imageChannelCount.read())
            
            print("The number of unique values in the image is {}".format(np.size(np.unique(arr_rs))))
            
            image = arr_rs
            import matplotlib.pyplot as plt
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
            
        if pPreviousRequest != None:
            pPreviousRequest.unlock()
        pPreviousRequest = pRequest
        #fi.imageRequestSingle()
        result = fi.acquisitionStop()
    else:
        print("imageRequestWaitFor failed (" + str(requestNr) + ", " + 
              acquire.ImpactAcquireException.getErrorCodeAsString(requestNr) + ")")
    result = fi.acquisitionStop()
    pDev.close()
    return True 


if __name__ == "__main__":
    acquire_frame(None)
    
    
# Saved for now:
# def __acquireExposure(self, exptime, fname):
#         #print("__acquireExposure: ",exptime,fname)
#         verbose = 0
#         exptime = min(max(exptime, 0.00001), 2.0)
#         devMgr = acquire.DeviceManager()
#         pDev = devMgr.getDevice(0)
#         conditionalSetProperty(pDev.interfaceLayout, acquire.dilGenICam)
#         conditionalSetProperty(pDev.acquisitionStartStopBehaviour, acquire.assbUser)
#         validValues = []
#         prop.getTranslationDictValues(validValues)
#         # #if pDev.isInUse:
#         # #    print(", !!!ALREADY IN USE!!!", end='')
#         pDev.open()
#         fi = acquire.FunctionInterface(pDev)
#         result = fi.acquisitionStop()
#         csbd = acquire.CameraSettingsBlueDevice(pDev)
#         csbd.expose_us.write(int(exptime * 1000000))
#         csbd.autoControlMode.write(0)
#         csbd.autoGainControl.write(0)
#         csbd.gain_dB.write(float(self.__gain))
#         csbd.pixelFormat.write(acquire.idpfMono12)
#         csbd.offset_pc.write(float(self.__percentOffset))
#         #dio = acquire.DigitalIOControl(pDev)
#         print(pDev.AnalogControl)
#         conditionalSetProperty(pDev.acquisitionStartStopBehaviour, acquire.assbUser)
#         #print("Turning off auto black level")
#         #ac.blackLevelAuto.write(0)
#         result = fi.acquisitionStart()
#         while fi.imageRequestSingle() == acquire.DMR_NO_ERROR:
#             if verbose: print(".",end='')
#         if verbose: print('')
#         pPreviousRequest = None
#         manuallyStartAcquisitionIfNeeded(pDev, fi)
#         requestNr = fi.imageRequestWaitFor(-1)
#         #fname = 'NULL'
#         if fi.isRequestNrValid(requestNr):
#             #print("RequestNrValid")
#             pRequest = fi.getRequest(requestNr)
#             if pRequest.isOK():
#                 #print("isOK")
#                 cbuf = (ctypes.c_char * pRequest.imageSize.read()).from_address(long(pRequest.imageData.read()))
#                 channelType = numpy.uint16 if pRequest.imageChannelBitDepth.read() > 8 else numpy.uint8
#                 arr = numpy.frombuffer(cbuf, dtype = channelType)
#                 arr.shape = (pRequest.imageHeight.read(), pRequest.imageWidth.read(), pRequest.imageChannelCount.read())
#                 img = Image.fromarray(arr[:,:,0])
#                 hdu = fits.PrimaryHDU(img)
#                 hdul = fits.HDUList([hdu])
#         #         #fname = dataDir + '/'+ prefix + '_' + filebase + '.fits'
#                 #print("Writing file: ",fname)
#                 hdul.writeto(fname,overwrite=True)
#             if pPreviousRequest != None:
#                 pPreviousRequest.unlock()
#             pPreviousRequest = pRequest
#             #fi.imageRequestSingle()
#             result = fi.acquisitionStop()
#         else:
#             print("imageRequestWaitFor failed (" + str(requestNr) + ", " + ImpactAcquireException.getErrorCodeAsString(requestNr) + ")")
#         result = fi.acquisitionStop()
#         pDev.close()
#         return 





