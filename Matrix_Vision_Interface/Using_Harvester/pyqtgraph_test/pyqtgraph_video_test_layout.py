#Matthew Leung
#March 2022

import numpy as np
from matrixvision_harvester_class import matrixvision_harvester_class as matrixvision_harvester_class
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
import pyqtgraph.ptime as ptime


app = QtGui.QApplication([])

view = pg.GraphicsView()
l = pg.GraphicsLayout(border=(100,100,100))
view.setCentralItem(l)
view.show()
view.setWindowTitle('Far Field Alignment')
#view.resize(800,600)
view.showMaximized()

#Title at top
text = "G-CLEF Fiber Lab - Far Field Alignment GUI"
title_obj = l.addLabel(text, col=0, colspan=2) #title_obj is of type pyqtgraph.graphicsItems.LabelItem.LabelItem
title_obj.setText(text, color="#EEEEEE", size="24pt", bold=True, italic=False)
l.nextRow()

#Add 2 plots into the first row (automatic position)
vb = l.addViewBox(lockAspect=True) #Box that allows internal scaling/panning of children by mouse drag
vb.setBackgroundColor(color="#0B0B0B")
img = pg.ImageItem(np.random.normal(size=(1104,1600)))
img.setOpts(border={'width':1})
vb.addItem(img)
vb.autoRange()
p2 = l.addPlot() #p2 is of type pyqtgraph.graphicsItems.PlotItem.PlotItem.PlotItem

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
p2.setLabel("bottom", "Time", **label_style)
p2.setLabel("left", "Eccentricity", **label_style)

p2.showGrid(x = True, y = True, alpha = 0.5) #add grid

###############################################################################

data1 = np.random.normal(size=300)
pen2 = pg.mkPen(color="#FF0000")
curve2 = p2.plot(data1, pen=pen2)
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

cti_filename = r"C:\Program Files\MATRIX VISION\mvIMPACT Acquire\bin\x64\mvGenTLProducer.cti"

camera_obj = matrixvision_harvester_class()
camera_obj.init_camera(cti_filename)
camera_obj.start_camera_acquisition(pixel_format='Mono8')
camera_obj.set_min_exposure_time(10)
camera_obj.set_exposure(exp_time=10)
camera_obj.set_frame_rate(5) #Set to 10 FPS

updateTime = ptime.time()
fps = 0

def updateData():
    global img, updateTime, fps

    data = camera_obj.get_snapshot_np_array()
    data = data.transpose(1, 0, 2)
    data = np.flip(data, 1)
    camera_obj.buffer.queue()
    del camera_obj.buffer

    ## Display the data
    img.setImage(data)
    update1()

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