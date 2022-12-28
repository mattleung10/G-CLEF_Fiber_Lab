#Matthew Leung
#Code last modified: March 15, 2022
"""
This script finds the contour bounding the fiber face, and fits an ellipse to
it, using the functions in near_field_edge_contours.py
Uses Harvester Python library to control camera and to acquire images.
Uses OpenCV imshow to display results.

THIS SCRIPT IS OLD. OPENCV IMSHOW LAGS. PYQT IS BETTER FOR LIVE VIDEO.
USE mv_harvester_class_ellipse_fit_pyqtgraph.py INSTEAD.
"""

import numpy as np
import cv2 as cv
from matrixvision_harvester_class import matrixvision_harvester_class as matrixvision_harvester_class
from near_field_edge_contours import find_edge_contours_ellipses, find_ellipse_major_axis_lines_points, find_ellipse_minor_axis_lines_points
import time


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

if __name__ == "__main__":
    cti_filename = r"C:\Program Files\MATRIX VISION\mvIMPACT Acquire\bin\x64\mvGenTLProducer.cti"
    
    camera_obj = matrixvision_harvester_class()
    camera_obj.init_camera(cti_filename)
    camera_obj.start_camera_acquisition(pixel_format='Mono8')
    camera_obj.set_min_exposure_time(10)
    camera_obj.set_exposure(exp_time=10)
    camera_obj.set_frame_rate(10) #Set to 10 FPS

    win_name = "Display"
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)

    while cv.waitKey(1) != 27:
        st = time.time()
        img = camera_obj.get_snapshot_np_array()
        camera_obj.buffer.queue()
        del camera_obj.buffer
        et = time.time()
        #print("Getting the image took {:.5f} seconds".format(et-st))
        
        #If needed, scale the plot image
        if camera_obj.bit_depth == 12:
            plot_image = (img / (2**12-1) * 255).astype('uint8') #convert to 8 bit image
        elif camera_obj.bit_depth == 8:
            plot_image = img
        
        stg = time.time()
        bit_depth = camera_obj.bit_depth
        st = time.time()
        ret = find_edge_contours_ellipses(img, bit_depth=bit_depth, verbosity=0)
        if ret is None:
            cv.imshow(win_name, plot_image)
            continue

        valid_contours, valid_ellipses, processed_ellipses, max_area_index = ret
        largest_ellipse = valid_ellipses[max_area_index]
        et = time.time()
        #print('Find ellipse took {:.5f} seconds'.format(et-st))
        #print(largest_ellipse)
        
        largest_processed_ellipse = processed_ellipses[max_area_index]
        x0, y0, a_double, b_double, angle, area, eccentricity2 = largest_processed_ellipse
        
        
        plot_text = "e = {:.5f}".format(np.sqrt(eccentricity2))
        
        if area > camera_obj.HEIGHT * camera_obj.WIDTH:
            cv.imshow(win_name, plot_image)
            continue
        plot_image = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
        plot_image = cv.ellipse(plot_image, largest_ellipse, (0,0,255), thickness=4)
        plot_image = cv.circle(plot_image, (int(x0),int(y0)), radius=10, color=(0, 0, 255), thickness=-1)
        xtop, ytop, xbot, ybot = find_ellipse_major_axis_lines_points(x0, y0, a_double, b_double, angle)
        drawline(plot_image,(xtop,ytop),(xbot,ybot),color=(0,0,255),thickness=2,gap=5)
        xtop, ytop, xbot, ybot = find_ellipse_minor_axis_lines_points(x0, y0, a_double, b_double, angle)
        drawline(plot_image,(xtop,ytop),(xbot,ybot),color=(0,0,255),thickness=2,gap=5)
        
        org = (int(camera_obj.WIDTH*0.02),int(camera_obj.HEIGHT*0.125))
        cv.putText(plot_image, plot_text, org=org, fontFace=cv.FONT_HERSHEY_DUPLEX, fontScale=4, color=(0,0,255))
        
        etg = time.time()
        #print("Processing the image took {:.5f} seconds".format(et-st))
        cv.imshow(win_name, plot_image)
        
    camera_obj.done_camera()
    print("Done!")