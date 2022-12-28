import numpy as np
import matplotlib.pyplot as plt

def plot_raw_image(filename):
     #https://stackoverflow.com/a/32189758
    fd = open(filename, 'rb')
    rows = 512
    cols = 512
    f = np.fromfile(fd, dtype=np.uint16,count=rows*cols)
    im = f.reshape((rows, cols))
    fd.close()
    
    plt.figure(dpi=300)
    plt.imshow(im, cmap='gray')
    plt.colorbar()
    #plt.savefig("ImageData1_plot.png", bbox_inches="tight")
    plt.show() 
    

if __name__ == "__main__":
    filename = 'ImageData1.raw'
    plot_raw_image(filename)
    