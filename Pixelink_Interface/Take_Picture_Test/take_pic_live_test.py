import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

from pixelinkWrapper import PxLApi
from getSnapshot_mod import get_snapshot

tempfilename = "temp.jpg"

# Not sure what it is for
def api_range_error(rc):
    return rc == PxLApi.ReturnCode.ApiInvalidParameterError or rc == PxLApi.ReturnCode.ApiOutOfRangeError

def set_exposure(hCamera, val):
    """
    Set the exposure in seconds
    """
    ret = PxLApi.getFeature(hCamera, PxLApi.FeatureId.EXPOSURE)
    if not(PxLApi.apiSuccess(ret[0])):
        print("!! Attempt to get exposure returned %i!" % ret[0])
        return
    
    params = ret[2]
    params[0] = val

    ret = PxLApi.setFeature(hCamera, PxLApi.FeatureId.EXPOSURE, PxLApi.FeatureFlags.MANUAL, params)
    if (not PxLApi.apiSuccess(ret[0])) and (not api_range_error(ret[0])):
        print("!! Attempt to set exposure returned %i!" % ret[0])
    
def save_image_from_camera():
    global tempfilename
    
    savefilename = tempfilename
    
    # Tell the camera we want to start using it.
	# NOTE: We're assuming there's only one camera.
    ret = PxLApi.initialize(0)
    if not PxLApi.apiSuccess(ret[0]):
        return 1
    hCamera = ret[1]
    
    set_exposure(hCamera, val=0.01)
    retVal = get_snapshot(hCamera, PxLApi.ImageFormat.JPEG, savefilename)
    
    PxLApi.uninitialize(hCamera) #Tell the camera we're done with it
    
def get_frame():
    global tempfilename
    save_image_from_camera()
    return cv2.imread(tempfilename, cv2.IMREAD_GRAYSCALE)
    
    
def main():
    pass

if __name__ == "__main__":
    main()