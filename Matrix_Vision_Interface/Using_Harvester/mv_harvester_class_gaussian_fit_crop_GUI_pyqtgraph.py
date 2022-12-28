#Matthew Leung
#Code last modified: March 16, 2022
"""
This script is used to focus the arms containing the Matrix Vision cameras.
Uses Harvester Python library to control camera and to acquire images.
Uses pyqtgraph to display results, which works better than matplotlib
interactive mode. pyqtgraph is built on PyQt, and works much better.

Running this script will give you a Qt GUI.

In this GUI, the user can select a custom ROI to fit a Gaussian function to
in realtime. Click the "Crop and Fit" checkbox to confirm the ROI and to start
the Gaussian fit.
This works better than iteratively finding the centroid of the image (as was
done in other scripts) and then cropping the image to a certain ROI. A user
selected ROI is more reliable and gives more consistent results.

This GUI also plots average FWHM (of the fitted Gaussian) VS time. This is 
used as a metric for focusing the spot.
"""

import numpy as np
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
import time
from Gaussian_Fit import crop_image_for_fit, fit_gaussian, gaussian, gaussian_FWHM
from matrixvision_harvester_class import matrixvision_harvester_class as matrixvision_harvester_class

def gaussian_fit_driver(cropped_img):
    #Fit Gaussian
    params, params_uncert, params_dict, params_uncert_dict, fit_fcn = fit_gaussian(cropped_img, clipped_gaussian=True)
    FWHM_x = gaussian_FWHM(params_dict['sigma_x'])
    FWHM_y = gaussian_FWHM(params_dict['sigma_y'])
    FWHM_avg = (FWHM_x + FWHM_y)/2
    x = np.arange(cropped_img.shape[1])
    y = np.arange(cropped_img.shape[0])
    X, Y = np.meshgrid(x, y)
    Z = gaussian(X, Y, *params)
    
    return Z, FWHM_x, FWHM_y, FWHM_avg

###############################################################################

def updateFWHMPlot(current_time, FWHM_avg):
    global time_array, FWHM_array, curve4, p24
    
    #Update time_array
    time_array = np.append(time_array, current_time)            
    time_array = time_array[1:]
    #Update eccentricity array
    FWHM_array = np.append(FWHM_array, FWHM_avg)
    FWHM_array = FWHM_array[1:]
    
    curve4.setData(x=time_array, y=FWHM_array)
    #curve2.setPos(time_array[ptr2], 0)
    p4.setRange(xRange=[np.min(time_array), np.max(time_array)])
    return True

def updateData():
    global img1, updateTime, fps, data, cropped_data, cb1

    data = camera_obj.get_snapshot_np_array()
    #No need to do the next two lines because I did pg.setConfigOptions(imageAxisOrder='row-major')
    #data = data.transpose(1, 0, 2)
    #data = np.flip(data, 1)
    data = data[:,:,0]
    camera_obj.buffer.queue()
    del camera_obj.buffer

    frame_time = time.time()
    ## Display the data
    img1.setImage(data)
    
    if cb1.isChecked():
        cropped_data = roi.getArrayRegion(data, img1)
        Z, FWHM_x, FWHM_y, FWHM_avg = gaussian_fit_driver(cropped_data)
        img3.setImage(Z)
        current_time = frame_time - start_time
        updateFWHMPlot(current_time, FWHM_avg)
    
    QtCore.QTimer.singleShot(1, updateData)
   
####################################################################################### 
   
if __name__ == "__main__":
    ###############################################################################
    
    cti_filename = r"C:\Program Files\MATRIX VISION\mvIMPACT Acquire\bin\x64\mvGenTLProducer.cti"
    
    camera_obj = matrixvision_harvester_class()
    camera_obj.init_camera(cti_filename)
    camera_obj.start_camera_acquisition(pixel_format='Mono8')
    camera_obj.set_min_exposure_time(10)
    camera_obj.set_exposure(exp_time=50)
    camera_obj.set_frame_rate(1) #Set to 10 FPS
    
    ###############################################################################
    
    app = QtGui.QApplication([])
    
    # Interpret image data as row-major instead of col-major
    pg.setConfigOptions(imageAxisOrder='row-major') #VERY IMPORTANT LINE!
    
    view = pg.GraphicsView()
    l = pg.GraphicsLayout(border=(100,100,100))
    view.setCentralItem(l)
    view.show()
    view.setWindowTitle('Focus Tool')
    #view.resize(800,600)
    view.showMaximized()
    
    #Title at top
    text = "G-CLEF Fiber Lab - Focus Tool"
    title_obj = l.addLabel(text, col=0, colspan=3) #title_obj is of type pyqtgraph.graphicsItems.LabelItem.LabelItem
    title_obj.setText(text, color="#EEEEEE", size="24pt", bold=True, italic=False)
    l.nextRow()
    
    ###############################################################################
    #Plot original image
    orig_width = camera_obj.WIDTH
    orig_height = camera_obj.HEIGHT
    p1 = l.addPlot(title="Original Image") #Instead of using ViewBox, use plot
    p1.setAspectLocked(lock=True, ratio=1) #LOCK ASPECT SO IMAGE IS DISPLAYED PROPERLY
    data = np.random.normal(size=(orig_width,orig_height))
    img1 = pg.ImageItem(data)
    img1.setOpts(border={'width':1})
    p1.addItem(img1)
    
    # Custom ROI for selecting an image region
    roi = pg.ROI(pos=[int(orig_width/4), int(orig_height/4)], size=[int(orig_width/2), int(orig_height/2)])
    roi.addScaleHandle([0.5, 1], [0.5, 0.5])
    roi.addScaleHandle([0, 0.5], [0.5, 0.5])
    p1.addItem(roi)
    roi.setZValue(10)  # make sure ROI is drawn above image
    
    ###############################################################################
    #Plot cropped image (cropped based on ROI)
    # p2 = l.addPlot(title="Cropped Image")
    # p2.setAspectLocked(lock=True, ratio=1) #LOCK ASPECT SO IMAGE IS DISPLAYED PROPERLY
    cropped_data = roi.getArrayRegion(data, img1)
    # img2 = pg.ImageItem(cropped_data)
    # img2.setOpts(border={'width':1})
    # p2.addItem(img2)
    
    ###############################################################################
    #Plot Gaussian fit of cropped image
    p3 = l.addPlot(title="Gaussain Fit of Crop Region")
    p3.setAspectLocked(lock=True, ratio=1) #LOCK ASPECT SO IMAGE IS DISPLAYED PROPERLY
    img3 = pg.ImageItem(np.random.normal(size=cropped_data.shape))
    img3.setOpts(border={'width':1})
    p3.addItem(img3)
    
    ###############################################################################
    
    p4 = l.addPlot() #p4 is of type pyqtgraph.graphicsItems.PlotItem.PlotItem.PlotItem
    #Change tick font size
    #https://stackoverflow.com/questions/41723497/how-to-change-ticks-fontsize-of-a-plot-using-pyqtgraph
    font=QtGui.QFont()
    font.setPixelSize(20)
    p4.getAxis("bottom").setStyle(tickFont = font)
    p4.getAxis("bottom").setStyle(tickTextOffset = 20)
    p4.getAxis("left").setStyle(tickFont = font)
    p4.getAxis("left").setStyle(tickTextOffset = 20)
    
    #Change label font size and color
    #https://stackoverflow.com/a/56904913
    label_style = {"color": "#EEEEEE", "font-size": "16pt"}
    p4.setTitle("Spot Average FWHM VS Time", color="#EEEEEE", size="18pt")
    p4.setLabel("bottom", "Time", **label_style)
    p4.setLabel("left", "Average FWHM", **label_style)
    
    p4.showGrid(x = True, y = True, alpha = 0.5) #add grid
    
    ###############################################################################
    #Plot time VS average FWHM
    prepoints = 100 #number of points before start of data acquisition
    time_array = np.linspace(-20,0,num=prepoints)
    FWHM_array = np.zeros(prepoints)
    pen4 = pg.mkPen(color="#FF0000")
    curve4 = p4.plot(x=time_array, y=FWHM_array, pen=pen4) #type pyqtgraph.graphicsItems.PlotDataItem.PlotDataItem
    
    ###############################################################################
    proxy1 = QtGui.QGraphicsProxyWidget()
    cb1 = QtGui.QCheckBox('Crop and Fit')
    proxy1.setWidget(cb1)
    l.nextRow()
    l2 = l.addLayout(colspan=3)
    l2.addItem(proxy1,row=0,col=0,colspan=1)
    cb1.setChecked(False)
    
    ###############################################################################
    # Monkey-patch the image to use our custom hover function. 
    # This is generally discouraged (you should subclass ImageItem instead),
    # but it works for a very simple use like this. 
    #img2.hoverEvent = imageHoverEvent

    start_time = time.time()
    updateData()
    
    ###############################################################################
    
    #Start Qt event loop unless running in interactive mode
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
        camera_obj.done_camera()
        print("Done!")
    