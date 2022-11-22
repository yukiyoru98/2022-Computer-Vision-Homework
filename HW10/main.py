import numpy as np
import cv2 as cv
import argparse
import math
'''
1.Laplacian x 2 (t=15)
2.Minimum-variance Laplacian (t=30)
3.Laplacian of Gaussian (t=3000)
4.Difference of Gaussian (inhibitory sigma = 1, excitatory sigma = 3, kernel size = 1, t=1)
'''

class Kernel:
    def __init__(self, array, origin):
        self.origin = origin
        self.array = array
        self.row = array.shape[0]
        self.col = array.shape[1]
    
    def pixel(self, y, x):
        return self.array[y, x]

    def Laplacian_4():    
        kernel_arr = np.array([ [ 0,  1, 0],
                                [ 1, -4, 1],
                                [ 0,  1, 0]], np.float32)
        
        return Kernel(kernel_arr, (1, 1))
    
    def Laplacian_8():    
        kernel_arr = np.array([ [ 1,  1, 1],
                                [ 1, -8, 1],
                                [ 1,  1, 1]], np.float32) / 3
        
        return Kernel(kernel_arr, (1, 1))

    def Minimum_variance_Laplacian():
        kernel_arr = np.array([ [  2, -1,  2],
                                [ -1, -4, -1],
                                [  2, -1,  2]], np.float32) / 3
        
        return Kernel(kernel_arr, (1, 1))

    
    def Laplacian_of_Gaussian():
        kernel_arr = np.array([ [ 0, 0, 0, -1, -1, -2, -1, -1, 0, 0, 0],
                                [ 0, 0, -2, -4, -8, -9, -8, -4, -2, 0, 0],
                                [ 0, -2, -7, -15, -22, -23, -22, -15, -7, -2, 0],
                                [ -1, -4, -15, -24, -14, -1, -14, -24, -15, -4, -1],
                                [ -1, -8, -22, -14, 52, 103, 52, -14, -22, -8, -1],
                                [ -2, -9, -23, -1, 103, 178, 103, -1, -23, -9, -2],
                                [ -1, -8, -22, -14, 52, 103, 52, -14, -22, -8, -1],
                                [ -1, -4, -15, -24, -14, -1, -14, -24, -15, -4, -1],
                                [ 0, -2, -7, -15, -22, -23, -22, -15, -7, -2, 0],
                                [ 0, 0, -2, -4, -8, -9, -8, -4, -2, 0, 0],
                                [ 0, 0, 0, -1, -1, -2, -1, -1, 0, 0, 0]], np.float32) 
                                
        return Kernel(kernel_arr, (5, 5))

    def Difference_of_Gaussian():
        kernel_arr = np.array([ [ -1, -3, -4, -6, -7, -8, -7, -6, -4, -3, -1],
                                [-3, -5, -8, -11, -13, -13, -13, -11, -8, -5, -3],
                                [-4, -8, -12, -16, -17, -17, -17, -16, -12, -8, -4],
                                [-6, -11, -16, -16, 0, 15, 0, -16, -16, -11, -6],
                                [-7, -13, -17, 0, 85, 160, 85, 0, -17, -13, -7],
                                [-8, -13, -17, 15, 160, 283, 160, 15, -17, -13, -8],
                                [-7, -13, -17, 0, 85, 160, 85, 0, -17, -13, -7],
                                [-6, -11, -16, -16, 0, 15, 0, -16, -16, -11, -6],
                                [-4, -8, -12, -16, -17, -17, -17, -16, -12, -8, -4],
                                [-3, -5, -8, -11, -13, -13, -13, -11, -8, -5, -3],
                                [ -1, -3, -4, -6, -7, -8, -7, -6, -4, -3, -1]], np.float32) 
                                
        return Kernel(kernel_arr, (5, 5))

def Convolution(source, i, j, kernel):
    conv = 0
    for k in range(0, kernel.row):
        for l in range(0, kernel.col):
            # kernel pixel displacement from kernel origin 
            # origin is determined by the first kernel since all kernels should have the same origin position
            dy = k - kernel.origin[0]
            dx = l - kernel.origin[1]
            # run all kernels
            conv += source[i+dy, j+dx] * kernel.pixel(k, l)

    return conv


def ZeroCrossingEdgeDetection(image, kernel, threshold):
    result = np.zeros(image.shape, np.float32)
    result[:] = 255 # initialize result image as all white
    row, col = image.shape
    pad_size = int(kernel.row / 2)
    padded_image = np.array(cv.copyMakeBorder(image, pad_size, pad_size, pad_size, pad_size, cv.BORDER_REPLICATE), dtype=np.float32)
    mask = np.zeros(image.shape, np.int8)

    for i in range(row):
        for j in range(col):
            mask[i, j] = CalculateLaplacian(padded_image, i, j, kernel, threshold)

    padded_mask = np.array(cv.copyMakeBorder(mask, 1, 1, 1, 1, cv.BORDER_REPLICATE), np.int8)
    for i in range(row):
        for j in range(col):
            if(padded_mask[i, j] != 1): # if mask value is 0 or -1, pixel remains white
                continue
            # if mask value is 1, check zero crossing
            isZeroCrossing = False
            # scan neighbors' mask value
            for dy in range(-1, 2, 1): # i-1, i, i+1
                for dx in range(-1, 2, 1): # j-1, j, j+1
                    neighbor_row = i + dy
                    neighbor_col = j + dx
                    if(padded_mask[neighbor_row, neighbor_col] == -1): # neighbor is -1, zero crossing
                        isZeroCrossing = True
                        result[i, j] = 0
                        break 
                if(isZeroCrossing):
                    break
    
    return result


def CalculateLaplacian(source, i, j, kernel, threshold):
    gradient_magnitude = Convolution(source, i, j, kernel)
    # compare with threshold
    if(gradient_magnitude >= threshold):
        return 1
    elif(gradient_magnitude <= -threshold):
        return -1
    else:
        return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source", default="lena.bmp")
    parser.add_argument("-m", "--method", default="L4")
    parser.add_argument("-t", "--threshold", default=15)
    args = parser.parse_args()
    
    threshold = int(args.threshold)
    # read image in grayscale mode
    source_image = cv.imread(args.source, 0)
    
    if(args.method == "L4"):
        # Laplacian 4
        laplacian_4 = ZeroCrossingEdgeDetection(source_image, Kernel.Laplacian_4(), threshold)
        cv.imwrite("laplacian_4.png", laplacian_4)

    elif(args.method == "L8"):
        # Laplacian 8
        laplacian_8 = ZeroCrossingEdgeDetection(source_image, Kernel.Laplacian_8(), threshold)
        cv.imwrite("laplacian_8.png", laplacian_8)

    elif(args.method == "MVL"):
        # Minimum Variance Laplacian
        minimum_variance_laplacian = ZeroCrossingEdgeDetection(source_image, Kernel.Minimum_variance_Laplacian(), threshold)
        cv.imwrite("MVL.png", minimum_variance_laplacian)

    elif(args.method == "LoG"):
        # Laplacian of Gaussian
        log = ZeroCrossingEdgeDetection(source_image, Kernel.Laplacian_of_Gaussian(), threshold)
        cv.imwrite("LoG.png", log)

    elif(args.method == "DoG"):
        # Difference of Gaussian
        dog = ZeroCrossingEdgeDetection(source_image, Kernel.Difference_of_Gaussian(), threshold)
        cv.imwrite("DoG.png", dog)


    
