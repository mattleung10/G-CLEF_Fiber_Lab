#Matthew Leung
#Code last updated: March 22, 2022
"""
Code to view TIF image
"""

import numpy as np
import matplotlib.pyplot as plt
import skimage.io
from PIL import Image

def read_uint12(data_chunk):
    data = np.frombuffer(data_chunk, dtype=np.uint8)
    fst_uint8, mid_uint8, lst_uint8 = np.reshape(data, (data.shape[0] // 3, 3)).astype(np.uint16).T
    fst_uint12 = (fst_uint8 << 4) + (mid_uint8 >> 4)
    snd_uint12 = ((mid_uint8 % 16) << 8) + lst_uint8
    return np.reshape(np.concatenate((fst_uint12[:, None], snd_uint12[:, None]), axis=1), 2 * fst_uint12.shape[0])

def open_raw_image(filename, width, height):
    #https://stackoverflow.com/a/32189758
    fd = open(filename, 'rb')
    
    #f = np.fromfile(fd, dtype=np.uint12,count=rows*cols)
    #image = f.reshape((rows, cols))
    #fd.close()
    #return image
    image = skimage.io.imread(filename)
    #image = Image.open(filename)
    #image = np.array(image)
    print(image)
    print("Maximum pixel value in image is {}".format(np.max(image)))
    print("Minimum pixel value in image is {}".format(np.min(image)))
    print("Number of unique values is {}".format(np.unique(image).size))
    #print(np.unique(image))
    image = image/16
    #print(image)
    
    plt.figure()
    plt.imshow(image, cmap='gray')
    plt.colorbar()
    plt.show()
    
    plt.figure()
    plt.hist(image.ravel(), 2**12)
    plt.title('Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.show()
    
if __name__ == "__main__":
    filename = r"C:\Users\fiberlab\Pictures\MatrixVision Images\20220317_1351 Near Field 200um Fiber.tif"
    #filename = r"C:\Users\fiberlab\Downloads\MVtest_001.tif"
    filename = r"C:\Users\fiberlab\Pictures\MatrixVision Images\20220317_1640 Near Field 200um Fiber 16 bit.tif"
    filename = r"C:\Users\fiberlab\Pictures\MatrixVision Images\20220318_1610 Near Field 200um Fiber 12 bit.tif"
    filename = r"C:\Users\fiberlab\Pictures\MatrixVision Images\20220321_1526 Near Field 200um Fiber.tif"
    filename = r"C:\Users\fiberlab\Pictures\MatrixVision Images\mono12p.tif"
    #filename = r"C:\Users\fiberlab\Pictures\MatrixVision Images\mono12.tif"
    #filename = r"C:\Users\fiberlab\Pictures\MatrixVision Images\20220322_1111 Near Field 200um Fiber.tif"
    filename = r"C:\Users\fiberlab\Pictures\MatrixVision Images\20220322_1114 Near Field 200um Fiber.tif"
    #filename = r"C:\Users\fiberlab\Pictures\MatrixVision Images\20220322_1116 Near Field 200um Fiber.tif"
    open_raw_image(filename, 1600, 1104)