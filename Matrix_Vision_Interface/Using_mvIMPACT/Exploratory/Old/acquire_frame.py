#Code last updated March 22, 2022
#Based on Peter Doherty's code

#! /usr/bin/env python 

# mvBlueCougar.py
#
# A 'camera' can be used to acquire images of various types: 'bias', 'dark'
# 'exp', and 'fe55'. It can also create a fake image file for testing.
#
from __future__ import print_function

# import standard packages
import sys
import os
import ctypes
import time
from datetime import date
import shutil
from PIL import Image
import numpy 

# Matrix Vision
from mvIMPACT import acquire
from mvIMPACT.Common import exampleHelper


def acquire_frame(filename, exptime=0.0, ampgain=0.0, verbosity=0): 
    #if self._verbose: self._log("Image acqisition start")
    exptime = min(max(exptime, 0.00001), 20.0)
    devMgr = acquire.DeviceManager()
    pDev = devMgr.getDevice(0)
    exampleHelper.conditionalSetProperty(pDev.interfaceLayout, acquire.dilGenICam)
    exampleHelper.conditionalSetProperty(pDev.acquisitionStartStopBehaviour, acquire.assbUser)
    pDev.open()
    fi = acquire.FunctionInterface(pDev)
    result = fi.acquisitionStop()
    #csbd = acquire.CameraSettingsBlueDevice(pDev)
    csbd = acquire.CameraSettingsBlueCOUGAR(pDev)
    csbd.expose_us.write(int(exptime * 1000000))
    csbd.autoControlMode.write(0)
    csbd.autoGainControl.write(0)
    csbd.gain_dB.write(float(ampgain)) #Electronic gain in dB
    csbd.pixelFormat.write(acquire.idpfMono12)
    #csbd.offset_pc.write(float(offset_pc)) #A float property defining the analogue sensor offset in percent of the allowed range
    #csbd.offset_pc.write(float(10))
    exampleHelper.conditionalSetProperty(pDev.acquisitionStartStopBehaviour, acquire.assbUser)
    result = fi.acquisitionStart()
    while fi.imageRequestSingle() == acquire.DMR_NO_ERROR:
        if verbosity > 0: print(".",end='')
    if verbosity > 0: print('')
    pPreviousRequest = None
    exampleHelper.manuallyStartAcquisitionIfNeeded(pDev, fi)
    requestNr = fi.imageRequestWaitFor(-1)
    if fi.isRequestNrValid(requestNr):
        pRequest = fi.getRequest(requestNr)
        if pRequest.isOK:
            pRequest.imageData.read()
            
            # python 2.7: 
            #cbuf = (ctypes.c_char * pRequest.imageSize.read()).from_address(long(pRequest.imageData.read()))
            # python 3.x:
            cbuf = (ctypes.c_char * pRequest.imageSize.read()).from_address(int(pRequest.imageData.read()))
            
            #channelType = numpy.uint16
            arr = numpy.frombuffer(cbuf, dtype = numpy.uint16)
            arr.shape = (pRequest.imageHeight.read(), pRequest.imageWidth.read(), pRequest.imageChannelCount.read())
            pixel_data = Image.fromarray(arr[:,:,0])
            
            print(pixel_data)
            
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

def ampgain(self, ampgain=None):
    if ampgain is not None:
        self._log("Setting gain to = %s" % ampgain)
        self._ampgain = float(ampgain)
    else:
        return self._ampgain

def percentOffset(self, percentOffset=None):
    if percentOffset is not None:
        self._log("Setting offset to = %s %" % percentOffset)
        self._percentOffset = float(percentOffset)
    else:
        return self._percentOffset


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





