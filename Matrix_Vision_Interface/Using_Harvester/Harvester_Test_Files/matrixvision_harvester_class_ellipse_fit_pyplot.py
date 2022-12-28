#Matthew Leung
#Code last updated: March 14, 2022

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matrixvision_harvester_class import matrixvision_harvester_class as matrixvision_harvester_class
from near_field_edge_contours import find_edge_contours_ellipses, find_ellipse_major_axis_lines_points, find_ellipse_minor_axis_lines_points

if __name__ == "__main__":
    cti_filename = r"C:\Program Files\MATRIX VISION\mvIMPACT Acquire\bin\x64\mvGenTLProducer.cti"
    
    camera_obj = matrixvision_harvester_class()
    camera_obj.init_camera(cti_filename)
    camera_obj.start_camera_acquisition(pixel_format='Mono8')
    camera_obj.set_min_exposure_time(10)
    camera_obj.set_exposure(exp_time=10)
    camera_obj.set_frame_rate(3)
    
    first_fit_done = False
    while True: #loop until we get an actual ellipse fit
        img = camera_obj.get_snapshot_np_array()
        camera_obj.buffer.queue()
        del camera_obj.buffer
        
        bit_depth = camera_obj.bit_depth
        ret = find_edge_contours_ellipses(img, bit_depth=bit_depth, verbosity=0)
        if ret is not None:
            break
    
    valid_contours, valid_ellipses, processed_ellipses, max_area_index = ret
    largest_ellipse = valid_ellipses[max_area_index]
    
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8,4), gridspec_kw={'width_ratios': [1, 1]})

    #Plot the original image
    im0 = ax[0].imshow(img, cmap='gray')
    ax[0].set_title('Fit')
    
    #Plot the cropped region
    xpoints = np.linspace(-10,0,num=20)
    ypoints = np.sin(xpoints)
    ax[1].plot(xpoints, ypoints)
    
    largest_processed_ellipse = processed_ellipses[max_area_index]
    x0, y0, a_double, b_double, angle, area, eccentricity2 = largest_processed_ellipse
    
    sc0 = ax[0].scatter(x0, y0, marker='o', s=20, color='r')
    ell = patches.Ellipse((x0, y0), a_double, b_double, angle, linewidth=1, edgecolor='r', facecolor='none')
    ax[0].add_patch(ell)
    plot_text = "$e$ = {:.5f}".format(np.sqrt(eccentricity2))
    
    xtop, ytop, xbot, ybot = find_ellipse_major_axis_lines_points(x0, y0, a_double, b_double, angle)
    pline_major,  = ax[0].plot([int(xtop), int(xbot)], [int(ytop), int(ybot)], color='r', linestyle='--', linewidth=1)
    xtop, ytop, xbot, ybot = find_ellipse_minor_axis_lines_points(x0, y0, a_double, b_double, angle)
    pline_minor,  = ax[0].plot([int(xtop), int(xbot)], [int(ytop), int(ybot)], color='r', linestyle='--', linewidth=1)
    
    ptext = ax[0].text(x=0.02,y=0.07, s=plot_text, fontsize=16, transform=ax[1].transAxes, color='r')

    plt.ion()
    try:
        while True:
            img = camera_obj.get_snapshot_np_array()            
            camera_obj.buffer.queue()
            del camera_obj.buffer
            
            #Update original image
            im0.set_data(img)
            #im0.set_clim(np.min(img), np.max(img))
            
            #plt.pause(0.05)
            #continue
            
            bit_depth = camera_obj.bit_depth
            ret = find_edge_contours_ellipses(img, bit_depth=bit_depth, verbosity=0)
            if ret is None:
                ptext.set_text('No valid output')
                continue
            
            valid_contours, valid_ellipses, processed_ellipses, max_area_index = ret
            largest_processed_ellipse = processed_ellipses[max_area_index]
            x0, y0, a_double, b_double, angle, area, eccentricity2 = largest_processed_ellipse
            sc0.set_offsets(np.array([x0, y0]))
            ell.set_center((x0,y0))
            ell.angle = angle
            ell.width = a_double
            ell.height = b_double
            
            plot_text = "$e$ = {:.5f}".format(np.sqrt(eccentricity2))
            ptext.set_text(plot_text)
            
            xtop, ytop, xbot, ybot = find_ellipse_major_axis_lines_points(x0, y0, a_double, b_double, angle)
            pline_major.set_xdata([int(xtop), int(xbot)])
            pline_major.set_ydata([int(ytop), int(ybot)])
            xtop, ytop, xbot, ybot = find_ellipse_minor_axis_lines_points(x0, y0, a_double, b_double, angle)
            pline_minor.set_xdata([int(xtop), int(xbot)])
            pline_minor.set_ydata([int(ytop), int(ybot)])
            
            plt.pause(0.05)
    except KeyboardInterrupt:
        pass
        
    plt.ioff()
    plt.show()
    
    camera_obj.done_camera()
    print("Done!")
