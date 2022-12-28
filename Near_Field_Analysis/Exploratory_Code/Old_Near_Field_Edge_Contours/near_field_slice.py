#Matthew Leung
#March 2022

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

def take_img_slice(img, x_slice_pos=None, y_slice_pos=None):
    """
    INPUTS:
        ::np.ndarray: img   #image
        ::int:: x_slice_pos #x coordinate of vertical slice
        ::int:: y_slice_pos #y coordinate of horizontal slice
    """
    h, w = img.shape
    
    if x_slice_pos is None:
        x_slice_pos = int(w/2)
    if y_slice_pos is None:
        y_slice_pos = int(h/2)
        
    hor_slice = img[y_slice_pos,:] #slice in horizontal direction
    vert_slice = img[:,x_slice_pos] #slice in vertical direction
    
    hor_points = np.arange(0,w)
    vert_points = np.arange(0,h)
    
    plt.figure()
    plt.imshow(img)
    plt.axhline(y_slice_pos, color='r') #horizontal slice
    plt.axvline(x_slice_pos, color='g') #vertical slice
    plt.show()
    
    plt.figure()
    plt.plot(hor_points, hor_slice, color='r')
    plt.title('Slice in horizontal direction')
    plt.xlabel('X pixel number')
    plt.ylabel('Pixel Value')
    plt.show()
    
    plt.figure()
    plt.plot(vert_points, vert_slice, color='g')
    plt.title('Slice in vertical direction')
    plt.xlabel('Y pixel number')
    plt.ylabel('Pixel Value')
    plt.show()

def view_gui(img, pix_max=None):
    h, w = img.shape #height and width of image
    
    #The following plotting method was modified from:
    #https://stackoverflow.com/questions/59144464/plotting-two-cross-section-intensity-at-the-same-time-in-one-figure/59147415#59147415
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    fig, ax = plt.subplots(figsize=(8,8))
    divider = make_axes_locatable(ax)
    top_ax = divider.append_axes("top", size=2, pad=0.1, sharex=ax) #Note: size and pad should be axes_grid.axes_size compatible
    top_ax.xaxis.set_tick_params(labelbottom=False) #don't show xlabel
    right_ax = divider.append_axes("right", size=2, pad=0.1, sharey=ax)
    right_ax.yaxis.set_tick_params(labelleft=False)
    
    top_ax.set_ylabel('Counts')
    right_ax.set_xlabel('Counts')
    
    ax.set_xlabel('Horizontal Pixel Number')
    ax.set_ylabel('Vertical Pixel Number')
    
    ax.imshow(img, cmap='gray', origin='lower', interpolation=None)
    
    ax.autoscale(enable=False)
    top_ax.autoscale(enable=False)
    right_ax.autoscale(enable=False)
    if pix_max is not None:
        top_ax.set_ylim(top=pix_max)
        right_ax.set_xlim(right=pix_max)
    else:
        top_ax.set_ylim(top=300)
        right_ax.set_xlim(right=300)
    h_line = ax.axhline(np.nan, color='r')
    h_prof, = top_ax.plot(np.arange(w),np.zeros(w), 'r-')
    v_line = ax.axvline(np.nan, color='g')
    v_prof, = right_ax.plot(np.zeros(h), np.arange(h), 'g-')
    
    def on_move(event):
        if event.inaxes is ax:
            cur_y = event.ydata
            cur_x = event.xdata
    
            h_line.set_ydata([cur_y,cur_y])
            h_prof.set_ydata(img[int(cur_y),:])
            v_line.set_xdata([cur_x,cur_x])
            v_prof.set_xdata(img[:,int(cur_x)])
    
            fig.canvas.draw_idle()
    
    fig.canvas.mpl_connect('motion_notify_event', on_move)
    
    plt.show()


if __name__ == "__main__":

    #filename = 'Images/220301_1634 NF of 200um Fiber, Laser Source.jpg'
    filename = 'Images/220301_1635 NF of 200um Fiber, Laser Source.jpg'
    #filename = 'Images/220301_1626 NF of 200um Fiber, Laser Source.jpg'
    
    img = cv.imread(filename, cv.IMREAD_GRAYSCALE)
    
    #take_img_slice(img)
    view_gui(img, pix_max=256*1.1)