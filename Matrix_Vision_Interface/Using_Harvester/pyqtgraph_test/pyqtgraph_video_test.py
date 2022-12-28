from matrixvision_harvester_class import matrixvision_harvester_class as matrixvision_harvester_class

"""
Demonstrates very basic use of ImageItem to display image data inside a ViewBox.
"""

from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
import pyqtgraph as pg
import pyqtgraph.ptime as ptime

app = QtGui.QApplication([])

## Create window with GraphicsView widget
win = pg.GraphicsLayoutWidget()
win.show()  ## show widget alone in its own window
win.setWindowTitle('pyqtgraph example: ImageItem')
win.ci.layout.setColumnStretchFactor(1, 2) #stretch 2nd column
view = win.addViewBox()

## lock the aspect ratio so pixels are always square
view.setAspectLocked(True)

## Create image item
img = pg.ImageItem(border='w')
view.addItem(img)

## Set initial view bounds
view.setRange(QtCore.QRectF(0, 0, 2000, 1104))

updateTime = ptime.time()
fps = 0

###############################################################################

p2 = win.addPlot()
data1 = np.random.normal(size=300)
curve2 = p2.plot(data1)
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

#shove all this stuff above into an init


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

## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
        camera_obj.done_camera()
        print("Done!")