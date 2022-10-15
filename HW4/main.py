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

def binarize(image, threshold): # input image should be gray scale
    row, col = image.shape
    result = np.zeros(image.shape, np.uint8)

    if(threshold < 0):  threshold = 0
    elif(threshold > 255): threshold = 255

    for i in range(row):
        for j in range(col):
            if(image[i, j] >= threshold):
                result[i, j] = 1
            else:
                result[i, j] = 0
    return result

def binary_dilation(image, kernel): #input should be binary image
    result = image.copy()
    row, col = image.shape
    # scan each pixel in image
    for i in range(row):
        for j in range(col):
            if(image[i, j] != 0): # if image pixel is white
                # scan each pixel in kernel
                for k in range(0, kernel.row):
                    for l in range(0, kernel.col):
                        # kernel pixel displacement from kernel origin
                        dy = k - kernel.origin[0]
                        dx = l - kernel.origin[1]
                        if(kernel.pixel(k, l) == 1): # if kernel pixel has value
                            if( (i + dy) >= 0 and (i + dy < row) and (j + dx) >= 0 and (j + dx) < col ): # if kernel pixel is within image
                                result[i+dy, j+dx] = 1 # dilate the corresponding pixel in result image

    return result

def binary_erosion(image, kernel): #input should be binary image
    result = np.zeros(image.shape)
    row, col = image.shape
    # scan each pixel in image
    for i in range(row):
        for j in range(col):
            # scan each pixel in kernel
            fit = True
            for k in range(0, kernel.row):
                if(not fit):    
                    break
                for l in range(kernel.col):
                    # kernel pixel displacement from kernel origin
                    dy = k - kernel.origin[0]
                    dx = l - kernel.origin[1]
                    if(kernel.pixel(k, l) == 1): 
                        if( (i + dy < 0) or (i + dy >= row) or (j + dx < 0) or (j + dx >= col) ): # if kernel pixel is out of image bound
                            result[i,j] = image[i,j]
                            fit = False
                            break
                        if(image[i+dy, j+dx] == 0): # if kernel pixel has value but corresponding image pixel does not
                            fit = False
                            break

            # if the entire kernel fits into the image
            if(fit):
                # preserve origin in result image
                result[i, j] = 1
    return result

def binary_complement(image):
    result = np.zeros(image.shape)
    row, col = image.shape
    for i in range(row):
        for j in range(col):
            result[i, j] = 1 - image[i, j]
    
    return result

def image_intersection(image_A, image_B):
    result = np.zeros(image_A.shape)
    row, col = image.shape
    for i in range(row):
        for j in range(col):
            if(image_A[i, j] == 1 and image_B[i, j] == 1):
                result[i, j] = 1
    
    return result

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source", default="lena.bmp")
    parser.add_argument("-b", "--binary_threshold", default=128)
    args = parser.parse_args()

    # read source image
    image = cv.imread(args.source, 0) # read with grayscale mode
    # binarize
    binary_image = binarize(image, threshold=int(args.binary_threshold))

    # setup octogonal kernel (3, 5, 5, 5, 3)
    oct_kernel_arr = np.array([ [0, 1, 1, 1, 0],
                                [1, 1, 1, 1, 1],
                                [1, 1, 1, 1, 1],
                                [1, 1, 1, 1, 1],
                                [0, 1, 1, 1, 0] ])
    oct_kernel = Kernel(oct_kernel_arr, (2, 2))

    # setup L kernels
    L_kernel_J_arr = np.array([ [0, 0, 0],
                                [1, 1, 0],
                                [0, 1, 0]])
    L_kernel_K_arr = np.array([ [0, 1, 1],
                                [0, 0, 1],
                                [0, 0, 0]])

    L_kernel_J = Kernel(L_kernel_J_arr, (1, 1))
    L_kernel_K = Kernel(L_kernel_K_arr, (1, 1))

    # binary dilation
    dilation_image = binary_dilation(binary_image, oct_kernel)
    plt.imsave('dilation.bmp', dilation_image, cmap='gray')

    # binary erosion
    erosion_image = binary_erosion(binary_image, oct_kernel)
    plt.imsave('erosion.bmp', erosion_image, cmap='gray')

    # opening
    opening_image = binary_dilation(erosion_image, oct_kernel)
    plt.imsave('opening.bmp', opening_image, cmap='gray')

    # closing 
    closing_image = binary_erosion(dilation_image, oct_kernel)
    plt.imsave('closing.bmp', closing_image, cmap='gray')

    # hit-and-miss
    complement = binary_complement(binary_image)
    hit_and_miss_image = image_intersection(binary_erosion(binary_image, L_kernel_J), binary_erosion(complement, L_kernel_K))
    plt.imsave('hit_and_miss.bmp', hit_and_miss_image, cmap='gray')