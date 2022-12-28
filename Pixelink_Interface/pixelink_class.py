#Matthew Leung
#Code last modified: February 15, 2022
"""
Class which wraps the routines in pixelinkWrapper API. Use this class for
image acquisition, to set the exposure, to save images, etc.
"""

import os
import numpy as np
import cv2 as cv
from pixelinkWrapper import PxLApi
from ctypes import create_string_buffer

SUCCESS = 0
FAILURE = 1

class pixelink_class:
    def __init__(self):
        self.hCamera = None #camera object
        
        #Image formats for Pixelink
        self.image_formats = {'jpg':PxLApi.ImageFormat.JPEG, 'bmp':PxLApi.ImageFormat.BMP,
                              'tiff':PxLApi.ImageFormat.TIFF, 'psd':PxLApi.ImageFormat.PSD,
                              'rgb24.bin':PxLApi.ImageFormat.RAW_BGR24, 'rgb24nondib.bin':PxLApi.ImageFormat.RAW_BGR24_NON_DIB,
                              'rgb48.bin':PxLApi.ImageFormat.RAW_RGB48, 'mono8.bin':PxLApi.ImageFormat.RAW_MONO8}
        return None
    
    def init_camera(self):
        # Tell the camera we want to start using it.
    	# NOTE: We're assuming there's only one camera.
        ret = PxLApi.initialize(0)
        if not PxLApi.apiSuccess(ret[0]): #Failure
            return False
        self.hCamera = ret[1]
        return True
    
    def done_camera(self):
        # Tell the camera we're done with it.
        PxLApi.uninitialize(self.hCamera)
        return True
    
    def set_exposure(self, exp_time):
        """
        Set the exposure in seconds
        """
        set_exposure(self.hCamera, exp_time)
    
    def get_snapshot_bytes(self, image_suffix):
        if image_suffix not in self.image_formats.keys():
            raise ValueError("Image suffix {} not supported".format(image_suffix))
        imageFormat = self.image_formats[image_suffix] #get the Pixelink ImageFormat corresponding to image_suffix
        
        return get_snapshot_bytes(self.hCamera, imageFormat)

    def get_snapshot_np_array(self, image_suffix):
        """
        INPUT:
            ::str:: image_suffix   #the encoding for the image (e.g. jpg, tiff)
        OUTPUT:
            ::np.ndarray:: decoded
        """
        formatedImage = self.get_snapshot_bytes(image_suffix)
        #######################################################################
        #formatedImage is a ctypes.char_Array_29594
        #We need to turn it to a np.ndarray
        #https://stackoverflow.com/questions/49511753/python-byte-image-to-numpy-array-using-opencv
        arr = np.frombuffer(formatedImage, np.uint8)                
        decoded = cv.imdecode(arr, cv.IMREAD_GRAYSCALE)
        #######################################################################
        return decoded
    
    def get_snapshot_and_save(self, image_suffix, filename):
        global SUCCESS
        
        #Check that the filename extension is consistent with image_suffix
        if os.path.basename(filename).split('.')[-1] != image_suffix:
            raise ValueError("filename extension does not match image_suffix")

        formatedImage = self.get_snapshot_bytes(image_suffix)
        r = save_image_to_file(filename, formatedImage)
        if r == SUCCESS:
            return True
        else:
            return False

#######################################################################################
#######################################################################################
 

def get_snapshot_bytes(hCamera, imageFormat):
    """
    Get a snapshot from the camera, and return a np.ndarray
    """
    global SUCCESS
    global FAILURE
    
    assert 0 != hCamera
    
    # Determine the size of buffer we'll need to hold an image from the camera
    import time
    #st = time.time()
    rawImageSize = determine_raw_image_size(hCamera)
    if 0 == rawImageSize:
        return FAILURE
    #et = time.time()
    #print("Getting image size took {} seconds".format(et-st))

    #st = time.time()
    # Create a buffer to hold the raw image
    rawImage = create_string_buffer(rawImageSize)
    #et = time.time()
    #print("Buffer creation took {} seconds".format(et-st))

    if 0 != len(rawImage):
        # Capture a raw image. The raw image buffer will contain image data on success. 
        
        st = time.time()
        ret = get_raw_image(hCamera, rawImage)
        et = time.time()
        print("get_raw_image took {} seconds".format(et-st))
        
        if PxLApi.apiSuccess(ret[0]):
            frameDescriptor = ret[1]
            
            assert 0 != len(rawImage)
            assert frameDescriptor
            #
            # Do any image processing here
            #
            
            #st = time.time()
            # Encode the raw image into something displayable
            ret = PxLApi.formatImage(rawImage, frameDescriptor, imageFormat)
            #et = time.time()
            #print("formatImage took {} seconds".format(et-st))
            if SUCCESS == ret[0]:
                formatedImage = ret[1]
                
                return formatedImage  
    return FAILURE

def determine_raw_image_size(hCamera):
    """
    Query the camera for region of interest (ROI), decimation, and pixel format
    Using this information, we can calculate the size of a raw image
    Returns 0 on failure
    """
    assert 0 != hCamera

    # Get region of interest (ROI)
    ret = PxLApi.getFeature(hCamera, PxLApi.FeatureId.ROI)
    params = ret[2]
    roiWidth = params[PxLApi.RoiParams.WIDTH]
    roiHeight = params[PxLApi.RoiParams.HEIGHT]

    # Query pixel addressing
        # assume no pixel addressing (in case it is not supported)
    pixelAddressingValueX = 1
    pixelAddressingValueY = 1

    ret = PxLApi.getFeature(hCamera, PxLApi.FeatureId.PIXEL_ADDRESSING)
    if PxLApi.apiSuccess(ret[0]):
        params = ret[2]
        if PxLApi.PixelAddressingParams.NUM_PARAMS == len(params):
            # Camera supports symmetric and asymmetric pixel addressing
            pixelAddressingValueX = params[PxLApi.PixelAddressingParams.X_VALUE]
            pixelAddressingValueY = params[PxLApi.PixelAddressingParams.Y_VALUE]
        else:
            # Camera supports only symmetric pixel addressing
            pixelAddressingValueX = params[PxLApi.PixelAddressingParams.VALUE]
            pixelAddressingValueY = params[PxLApi.PixelAddressingParams.VALUE]

    # We can calulate the number of pixels now.
    numPixels = (roiWidth / pixelAddressingValueX) * (roiHeight / pixelAddressingValueY)
    ret = PxLApi.getFeature(hCamera, PxLApi.FeatureId.PIXEL_FORMAT)

    # Knowing pixel format means we can determine how many bytes per pixel.
    params = ret[2]
    pixelFormat = int(params[0])

    # And now the size of the frame
    pixelSize = PxLApi.getBytesPerPixel(pixelFormat)

    return int(numPixels * pixelSize)


def get_raw_image(hCamera, rawImage):
    """
    Capture an image from the camera.
     
    NOTE: PxLApi.getNextFrame is a blocking call. 
    i.e. PxLApi.getNextFrame won't return until an image is captured.
    So, if you're using hardware triggering, it won't return until the camera is triggered.
    Returns a return code with success and frame descriptor information or API error
    """
    global FAILURE
    
    assert 0 != hCamera
    assert 0 != len(rawImage)

    MAX_NUM_TRIES = 4

    # Put camera into streaming state so we can capture an image
    ret = PxLApi.setStreamState(hCamera, PxLApi.StreamState.START)
    if not PxLApi.apiSuccess(ret[0]):
        return FAILURE
      
    # Get an image
    # NOTE: PxLApi.getNextFrame can return ApiCameraTimeoutError on occasion.
    # How you handle this depends on your situation and how you use your camera. 
    # For this sample app, we'll just retry a few times.
    ret = (PxLApi.ReturnCode.ApiUnknownError,)

    for i in range(MAX_NUM_TRIES):
        ret = PxLApi.getNextFrame(hCamera, rawImage)
        if PxLApi.apiSuccess(ret[0]):
            break

    # Done capturing, so no longer need the camera streaming images.
    # Note: If ret is used for this call, it will lose frame descriptor information.
    PxLApi.setStreamState(hCamera, PxLApi.StreamState.STOP)

    return ret


def save_image_to_file(fileName, formatedImage):
    """
    Save the encoded image buffer to a file
    This overwrites any existing file
    Returns SUCCESS or FAILURE
    """
    #THIS FUNCTION HAS BEEN MODIFIED TO SAVE THE IMAGE IN ANY fileName
    
    global SUCCESS
    global FAILURE
    
    assert fileName
    assert 0 != len(formatedImage)

    filepath = fileName
    # Open a file for binary write
    file = open(filepath, "wb")
    if None == file:
        return FAILURE
    numBytesWritten = file.write(formatedImage)
    file.close()

    if numBytesWritten == len(formatedImage):
        return SUCCESS

    return FAILURE

#######################################################################################
#######################################################################################
    
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
        
