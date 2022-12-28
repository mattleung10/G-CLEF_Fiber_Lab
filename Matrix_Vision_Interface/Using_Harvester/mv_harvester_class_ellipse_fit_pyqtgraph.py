#Matthew Leung
#Code last modified: March 16, 2022
"""
This script is used to align and focus the FAR FIELD arm, assuming a circular fiber.
Uses Harvester Python library to control camera and to acquire images.
Uses pyqtgraph to display results.

Finds the eccentricity of the far field image, assuming a circular fiber.
In the ideal case, the eccentricity should be 0, since the far field image
should be circular. However, due to misalignment, the far field image of a
circular fiber can be elliptical.

Uses the procedure in near_field_edge_contours.py:
    1) Gaussian blur the image to smooth out noise
    2) Threshold the image using Otsu's Method
    3) Find contours of thresholded image
    4) Fit an ellipse to each contour
    5) Find the eccentricity and area of each ellipse
    6) Find the ellipse with the largest area, and take that as the ellipse
       which encloses the circular fiber face

Running this script will give you a Qt GUI.

This GUI plots eccentricity as a function of time. Eccentricity of the circular
fiber image is a metric which could be used for alignment/focus.
"""

import numpy as np
import cv2 as cv
from matrixvision_harvester_class import matrixvision_harvester_class as matrixvision_harvester_class
from near_field_edge_contours import find_edge_contours_ellipses, find_ellipse_major_axis_lines_points, find_ellipse_minor_axis_lines_points
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
import time

def process_image(image):
    global camera_obj

    #If needed, scale the plot image
    if camera_obj.bit_depth == 12:
        plot_image = (image / (2**12-1) * 255).astype('uint8') #convert to 8 bit image
    elif camera_obj.bit_depth == 8:
        plot_image = image
    
    bit_depth = camera_obj.bit_depth
    ret = find_edge_contours_ellipses(image, bit_depth=bit_depth, verbosity=0)
    if ret is None:
        return [image, None]

    valid_contours, valid_ellipses, processed_ellipses, max_area_index = ret
    largest_ellipse = valid_ellipses[max_area_index]
    
    largest_processed_ellipse = processed_ellipses[max_area_index]
    x0, y0, a_double, b_double, angle, area, eccentricity2 = largest_processed_ellipse
    
    #plot_text = "e = {:.5f}".format(np.sqrt(eccentricity2))
    
    if area > camera_obj.HEIGHT * camera_obj.WIDTH:
        return [image, None]

    plot_image = cv.cvtColor(image, cv.COLOR_GRAY2RGB)
    plot_image = cv.ellipse(plot_image, largest_ellipse, (255,0,0), thickness=4)
    plot_image = cv.circle(plot_image, (int(x0),int(y0)), radius=10, color=(255, 0, 0), thickness=-1)
    xtop, ytop, xbot, ybot = find_ellipse_major_axis_lines_points(x0, y0, a_double, b_double, angle)
    drawline(plot_image,(xtop,ytop),(xbot,ybot),color=(255,0,0),thickness=2,gap=5)
    xtop, ytop, xbot, ybot = find_ellipse_minor_axis_lines_points(x0, y0, a_double, b_double, angle)
    drawline(plot_image,(xtop,ytop),(xbot,ybot),color=(255,0,0),thickness=2,gap=5)
    
    #org = (int(camera_obj.WIDTH*0.02),int(camera_obj.HEIGHT*0.125))
    #cv.putText(plot_image, plot_text, org=org, fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=4, color=(255,0,0))
    
    return plot_image, np.sqrt(eccentricity2)


def drawline(img,pt1,pt2,color,thickness=1,style='dotted',gap=20):
    #From https://stackoverflow.com/a/26711359
    dist =((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)**.5
    pts= []
    for i in  np.arange(0,dist,gap):
        r=i/dist
        x=int((pt1[0]*(1-r)+pt2[0]*r)+.5)
        y=int((pt1[1]*(1-r)+pt2[1]*r)+.5)
        p = (x,y)
        pts.append(p)

    if style=='dotted':
        for p in pts:
            cv.circle(img,p,thickness,color,-1)
    else:
        s=pts[0]
        e=pts[0]
        i=0
        for p in pts:
            s=e
            e=p
            if i%2==1:
                cv.line(img,s,e,color,thickness)
            i+=1

###############################################################################            

def updateEccentricityPlot(current_time, eccentricity):
    global time_array, e_array, curve2, ptr2, p2
    
    #Update time_array
    time_array = np.append(time_array, current_time)            
    time_array = time_array[1:]
    #Update eccentricity array
    e_array = np.append(e_array, eccentricity)
    e_array = e_array[1:]
    
    ptr2 += 1
    curve2.setData(x=time_array, y=e_array)
    #curve2.setPos(time_array[ptr2], 0)
    p2.setRange(xRange=[np.min(time_array), np.max(time_array)])

def updateData():
    global camera_obj, img, frame_time, e_text

    raw_image = camera_obj.get_snapshot_np_array()
    camera_obj.buffer.queue()
    del camera_obj.buffer

    frame_time = time.time()
    
    plot_image, eccentricity = process_image(raw_image)
    
    #PyQtGraph interprets the axes of image data as [width, height], which is the opposite of most other standards
    plot_image = plot_image.transpose(1, 0, 2)
    plot_image = np.flip(plot_image, 1)
    
    ## Display the data
    img.setImage(plot_image)
    
    current_time = frame_time - start_time
    if eccentricity is not None:
        updateEccentricityPlot(current_time, eccentricity)
        e_text.setText("{:.5f}".format(eccentricity))
    else:
        updateEccentricityPlot(current_time, e_array[-1])
        e_text.setText("?")

    QtCore.QTimer.singleShot(1, updateData)

if __name__ == "__main__":
    app = QtGui.QApplication([])
    
    view = pg.GraphicsView()
    l = pg.GraphicsLayout(border=(100,100,100))
    view.setCentralItem(l)
    view.show()
    view.setWindowTitle('Far Field Alignment')
    #view.resize(800,600)
    view.showMaximized() #Maximize window
    
    #Title at top
    text = "G-CLEF Fiber Lab - Far Field Alignment GUI"
    title_obj = l.addLabel(text, col=0, colspan=3) #title_obj is of type pyqtgraph.graphicsItems.LabelItem.LabelItem
    title_obj.setText(text, color="#EEEEEE", size="24pt", bold=True, italic=False)
    l.nextRow()
    
    l2 = l.addLayout(colspan=3, rowspan=4, border=(50,0,0))
    
    ###############################################################################
    #Create plot to disply the eccentricity value
    e_text_graph = l2.addPlot(row=0, col=0, rowspan=1, colspan=1)
    e_text_graph.setTitle("Eccentricity", color="#EEEEEE", size="18pt")
    e_text_graph.hideAxis('bottom')
    e_text_graph.hideAxis('left')
    e_text = pg.TextItem("test", anchor=(0.5, 0.5), color="w")
    font=QtGui.QFont()
    font.setPixelSize(100)
    e_text.setFont(font)
    e_text_graph.addItem(e_text)
    
    ###############################################################################
    #vb = l.addViewBox(lockAspect=True) #Box that allows internal scaling/panning of children by mouse drag
    vb = l2.addViewBox(lockAspect=True, row=1, col=0, rowspan=3)
    vb.setBackgroundColor(color="#0B0B0B") #set background color of ViewBox
    img = pg.ImageItem(np.random.normal(size=(1104,1600)))
    img.setOpts(border={'width':1}) #add border to ImageItem
    vb.addItem(img)
    vb.autoRange()
    
    ###############################################################################
    #Do this so that the ratio of the text plot to the image viewbox is 1:4
    #Hack: https://stackoverflow.com/a/66299913
    px1 = l2.addLabel(text='',row=0, col=2)
    px2 = l2.addLabel(text='',row=1, col=2)
    px3 = l2.addLabel(text='',row=2, col=2)
    px3 = l2.addLabel(text='',row=3, col=2)
    
    ###############################################################################
    #Eccentricity VS Time Plot
    #p2 = l.addPlot() #p2 is of type pyqtgraph.graphicsItems.PlotItem.PlotItem.PlotItem
    p2 = l2.addPlot(row=0, col=1,rowspan=4)
    
    #Change tick font size
    #https://stackoverflow.com/questions/41723497/how-to-change-ticks-fontsize-of-a-plot-using-pyqtgraph
    font=QtGui.QFont()
    font.setPixelSize(20)
    p2.getAxis("bottom").setStyle(tickFont = font)
    p2.getAxis("bottom").setStyle(tickTextOffset = 20)
    p2.getAxis("left").setStyle(tickFont = font)
    p2.getAxis("left").setStyle(tickTextOffset = 20)
    
    #Change label font size and color
    #https://stackoverflow.com/a/56904913
    label_style = {"color": "#EEEEEE", "font-size": "16pt"}
    p2.setTitle("Far Field Beam Eccentricity VS Time", color="#EEEEEE", size="18pt")
    p2.setLabel("bottom", "Time Elapsed [Seconds]", **label_style)
    p2.setLabel("left", "Eccentricity", **label_style)
    p2.showGrid(x = True, y = True, alpha = 0.5) #add grid
    p2.setRange(yRange=[0, 1.05])
    
    ###############################################################################
    
    #Plot time VS average FWHM
    prepoints = 100 #number of points before start of data acquisition
    time_array = np.linspace(-20,0,num=prepoints)
    e_array = np.zeros(prepoints)
    pen2 = pg.mkPen(color="#FF0000")
    curve2 = p2.plot(x=time_array, y=e_array, pen=pen2) #type pyqtgraph.graphicsItems.PlotDataItem.PlotDataItem

    ###########################################################################
    cti_filename = r"C:\Program Files\MATRIX VISION\mvIMPACT Acquire\bin\x64\mvGenTLProducer.cti"
    camera_obj = matrixvision_harvester_class()
    camera_obj.init_camera(cti_filename)
    camera_obj.start_camera_acquisition(pixel_format='Mono8')
    camera_obj.set_min_exposure_time(10)
    camera_obj.set_exposure(exp_time=10)
    camera_obj.set_frame_rate(3) #Set to low FPS
    ###########################################################################

    start_time = time.time()
    ptr2 = 0
    
    updateData()
    
    #Start Qt event loop unless running in interactive mode
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
        camera_obj.done_camera()
        print("Done!")
    