#Matthew Leung
#March 2022

import numpy as np
from matrixvision_harvester_class import matrixvision_harvester_class as matrixvision_harvester_class
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
import pyqtgraph.ptime as ptime

###############################################################################

cti_filename = r"C:\Program Files\MATRIX VISION\mvIMPACT Acquire\bin\x64\mvGenTLProducer.cti"

camera_obj = matrixvision_harvester_class()
camera_obj.init_camera(cti_filename)
camera_obj.start_camera_acquisition(pixel_format='Mono8')
camera_obj.set_min_exposure_time(10)
camera_obj.set_exposure(exp_time=10)
camera_obj.set_frame_rate(5) #Set to 10 FPS

###############################################################################

app = QtGui.QApplication([])

# Interpret image data as row-major instead of col-major
pg.setConfigOptions(imageAxisOrder='row-major') #VERY IMPORTANT LINE!

view = pg.GraphicsView()
l = pg.GraphicsLayout(border=(100,100,100))
view.setCentralItem(l)
view.show()
view.setWindowTitle('Far Field Alignment')
#view.resize(800,600)
view.showMaximized()

#Title at top
text = "G-CLEF Fiber Lab - Far Field Alignment GUI"
title_obj = l.addLabel(text, col=0, colspan=3) #title_obj is of type pyqtgraph.graphicsItems.LabelItem.LabelItem
title_obj.setText(text, color="#EEEEEE", size="24pt", bold=True, italic=False)
l.nextRow()

orig_width = camera_obj.WIDTH
orig_height = camera_obj.HEIGHT
#Add 2 plots into the first row (automatic position)
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

p2 = l.addPlot(title="Cropped Image")
p2.setAspectLocked(lock=True, ratio=1) #LOCK ASPECT SO IMAGE IS DISPLAYED PROPERLY
cropped_data = roi.getArrayRegion(data, img1)
img2 = pg.ImageItem(cropped_data)
img2.setOpts(border={'width':1})
p2.addItem(img2)


p3 = l.addPlot() #p3 is of type pyqtgraph.graphicsItems.PlotItem.PlotItem.PlotItem
#Change tick font size
#https://stackoverflow.com/questions/41723497/how-to-change-ticks-fontsize-of-a-plot-using-pyqtgraph
font=QtGui.QFont()
font.setPixelSize(20)
p3.getAxis("bottom").setStyle(tickFont = font)
p3.getAxis("bottom").setStyle(tickTextOffset = 20)
p3.getAxis("left").setStyle(tickFont = font)
p3.getAxis("left").setStyle(tickTextOffset = 20)

#Change label font size and color
#https://stackoverflow.com/a/56904913
label_style = {"color": "#EEEEEE", "font-size": "16pt"}
p3.setTitle("Far Field Beam Eccentricity VS Time", color="#EEEEEE", size="18pt")
p3.setLabel("bottom", "Time", **label_style)
p3.setLabel("left", "Eccentricity", **label_style)

p3.showGrid(x = True, y = True, alpha = 0.5) #add grid

proxy = QtGui.QGraphicsProxyWidget()
cb = QtGui.QCheckBox('Crop and Fit')
proxy.setWidget(cb)
l.nextRow()
l2 = l.addLayout(rowspan=3)
l2.addItem(proxy,row=0,col=0)

###############################################################################

data1 = np.random.normal(size=300)
pen2 = pg.mkPen(color="#FF0000")
curve2 = p3.plot(data1, pen=pen2)
ptr1 = 0
def update1():
    global data1, ptr1
    data1[:-1] = data1[1:]  # shift data in the array one sample left
                            # (see also: np.roll)
    data1[-1] = np.random.normal()
    
    ptr1 += 1
    curve2.setData(data1)
    curve2.setPos(ptr1, 0)

###############################################################################

updateTime = ptime.time()
fps = 0

def imageHoverEvent(event):
    """Show the position, pixel, and value under the mouse cursor.
    """
    global img2, cropped_data
    if event.isExit():
        p2.setTitle("")
        return
    pos = event.pos()
    i, j = pos.y(), pos.x()
    i = int(np.clip(i, 0, cropped_data.shape[0] - 1))
    j = int(np.clip(j, 0, cropped_data.shape[1] - 1))
    val = cropped_data[i, j]
    ppos = img2.mapToParent(pos)
    x, y = ppos.x(), ppos.y()
    p2.setTitle("pos: (%0.1f, %0.1f)  pixel: (%d, %d)  value: %g" % (x, y, i, j, val))

# Monkey-patch the image to use our custom hover function. 
# This is generally discouraged (you should subclass ImageItem instead),
# but it works for a very simple use like this. 
img2.hoverEvent = imageHoverEvent



def updateData():
    global img1, updateTime, fps, data, cropped_data

    data = camera_obj.get_snapshot_np_array()
    #data = data.transpose(1, 0, 2)
    #data = np.flip(data, 1)
    data = data[:,:,0]
    camera_obj.buffer.queue()
    del camera_obj.buffer

    ## Display the data
    img1.setImage(data)
    update1()

    cropped_data = roi.getArrayRegion(data, img1)
    img2.setImage(cropped_data)

    QtCore.QTimer.singleShot(1, updateData)
    now = ptime.time()
    fps2 = 1.0 / (now-updateTime)
    updateTime = now
    fps = fps * 0.9 + fps2 * 0.1
    
updateData()


if __name__ == '__main__':
    
    #Start Qt event loop unless running in interactive mode
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
        camera_obj.done_camera()
        print("Done!")