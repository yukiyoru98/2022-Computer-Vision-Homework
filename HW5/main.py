import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import argparse

class Kernel:
    def __init__(self, array, origin):
        self.origin = origin
        self.array = array
        self.row = array.shape[0]
        self.col = array.shape[1]
    
    def pixel(self, y, x):
        return self.array[y, x]

def grayscale_dilation(image, kernel): #input should be binary image, flat kernel only
    result = image.copy()
    row, col = image.shape
    # scan each pixel in image
    for i in range(row):
        for j in range(col):
            local_max = 0
            # scan each pixel in kernel
            for k in range(0, kernel.row):
                for l in range(0, kernel.col):
                    # kernel pixel displacement from kernel origin
                    dy = k - kernel.origin[0]
                    dx = l - kernel.origin[1]
                    if(kernel.pixel(k, l) == 1): # if kernel pixel has value
                        if( (i + dy) >= 0 and (i + dy < row) and (j + dx) >= 0 and (j + dx) < col ): # if kernel pixel is within image
                            if( local_max < image[i+dy, j+dx]): # record local maximum
                                local_max = image[i+dy, j+dx] 
            # dilate the corresponding pixel in result image
            result[i, j] = local_max

    return result

                

def grayscale_erosion(image, kernel): #input should be binary image, flat kernel only
    result = image.copy()
    row, col = image.shape
    # scan each pixel in image
    for i in range(row):
        for j in range(col):
            local_min = 255
            # scan each pixel in kernel
            for k in range(0, kernel.row):
                for l in range(0, kernel.col):
                    # kernel pixel displacement from kernel origin
                    dy = k - kernel.origin[0]
                    dx = l - kernel.origin[1]
                    if(kernel.pixel(k, l) == 1): # if kernel pixel has value
                        if( (i + dy) >= 0 and (i + dy < row) and (j + dx) >= 0 and (j + dx) < col ): # if kernel pixel is within image
                            if( local_min > image[i+dy, j+dx]): # record local minimum 
                                local_min = image[i+dy, j+dx] 
            # erode the corresponding pixel in result image
            result[i, j] = local_min

    return result


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source", default="lena.bmp")
    args = parser.parse_args()

    # read source image
    image = cv.imread(args.source, 0) # read with grayscale mode

    # setup flat octogonal kernel (3, 5, 5, 5, 3)
    oct_kernel_arr = np.array([ [0, 1, 1, 1, 0],
                                [1, 1, 1, 1, 1],
                                [1, 1, 1, 1, 1],
                                [1, 1, 1, 1, 1],
                                [0, 1, 1, 1, 0] ])
    oct_kernel = Kernel(oct_kernel_arr, (2, 2))

    # grayscale dilation
    dilation_image = grayscale_dilation(image, oct_kernel)
    plt.imsave('dilation.bmp', dilation_image, cmap='gray')

    # grayscale erosion
    erosion_image = grayscale_erosion(image, oct_kernel)
    plt.imsave('erosion.bmp', erosion_image, cmap='gray')

    # opening
    opening_image = grayscale_dilation(erosion_image, oct_kernel)
    plt.imsave('opening.bmp', opening_image, cmap='gray')

    # closing 
    closing_image = grayscale_erosion(dilation_image, oct_kernel)
    plt.imsave('closing.bmp', closing_image, cmap='gray')
