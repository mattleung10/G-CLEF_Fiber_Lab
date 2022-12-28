#Matthew Leung
#Code last modified: February 11, 2022
"""
This script is a test of the class pixelink_class.
Takes an image using pixelink_class and displays it.
"""

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pixelink_class import pixelink_class as pixelink_class

if __name__ == "__main__":
    camera_obj = pixelink_class()
    camera_obj.init_camera()
    camera_obj.set_exposure(exp_time=0.01)
    img = camera_obj.get_snapshot_np_array('jpg')
    print(img.shape)
    fig, ax = plt.subplots()
    im = ax.imshow(img)
    fig.colorbar(im)
    
    # def update(i):
    #     im.set_data(camera_obj.get_snapshot_np_array('jpg'))
    
    # ani = FuncAnimation(plt.gcf(), update, interval=200)
    plt.show()
        
    '''
    plt.ion()
    for i in range(0,100,1):
        img = camera_obj.get_snapshot_np_array('jpg')
        im.set_data(img)
        plt.pause(0.2)
    
    plt.ioff() # due to infinite loop, this gets never called.
    plt.show()
    '''
    
    
    camera_obj.done_camera() #You have to do this! Otherwise the script crashes on next run
        
    
    #plt_obj = plt.imshow(img)
    #plt.colorbar()
    #plt.show()



    