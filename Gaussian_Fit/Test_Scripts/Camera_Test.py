#Matthew Leung
#February 14, 2022
"""
This script is just a test to acquire video using OpenCV VideoCapture.
"""

import matplotlib.pyplot as plt
import cv2 as cv

if __name__ == "__main__":
    cap = cv.VideoCapture(0)
    if cap.isOpened() != True:
        print("Camera cannot be opened")
        exit()
        
    def get_data():
        #take each frame
        ret, frame = cap.read()
        return 255 - cv.cvtColor(frame, cv.COLOR_BGR2GRAY) #invert image
    
    fig, ax = plt.subplots()
    im = ax.imshow(get_data(), cmap='gray')
    fig.colorbar(im)
    
    plt.ion()
    for i in range(0,1000,1):
        im.set_data(get_data())
        plt.pause(0.05)
    
    plt.ioff() # due to infinite loop, this gets never called.
    plt.show()
    
    cap.release()

