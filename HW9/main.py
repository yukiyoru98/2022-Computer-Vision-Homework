import numpy as np
import cv2 as cv
import argparse
import math
'''
1.Roberts operator
2.Prewitt edge detector
3.Sobel edge detector
4.Frei and Chen gradient operator
5.Kirsch compass operator
6.Robinson compass operator
7.Nevatia-Babu 5x5 operator
'''

class Kernel:
    def __init__(self, array, origin):
        self.origin = origin
        self.array = array
        self.row = array.shape[0]
        self.col = array.shape[1]
    
    def pixel(self, y, x):
        return self.array[y, x]

def RobertsOperator(image, threshold):
    result = np.zeros(image.shape, np.float32)
    row, col = image.shape
    padded_image = np.array(cv.copyMakeBorder(image, 1, 1, 1, 1, cv.BORDER_CONSTANT, value=0), dtype=np.float32)
    # scan each pixel 
    for i in range(row):
        for j in range(col):
            r1 = -padded_image[i, j] + padded_image[i+1, j+1]
            r2 = -padded_image[i, j+1] + padded_image[i+1, j]
            gradient_magnitude = np.linalg.norm([r1, r2])
            # gradient_magnitude = math.sqrt(math.pow(r1, 2) + math.pow(r2, 2))
            if(gradient_magnitude < threshold):
                result[i, j] = 255
    
    return result

def PrewittEdgeDetector(image, threshold):
    result = np.zeros(image.shape, np.float32)
    row, col = image.shape
    padded_image = np.array(cv.copyMakeBorder(image, 1, 1, 1, 1, cv.BORDER_REPLICATE), dtype=np.float32)
  
    kernel_1_arr = np.array([   [-1, -1, -1],
                                [0, 0, 0],
                                [1, 1, 1]])
    kernel_2_arr = np.transpose(kernel_1_arr)

    kernel_1 = Kernel(kernel_1_arr, (1, 1))
    kernel_2 = Kernel(kernel_2_arr, (1, 1))

    # scan each pixel 
    for i in range(row):
        for j in range(col):
            p1 = p2 = 0
            # run kernel over pixel
            for k in range(0, kernel_1.row):
                for l in range(0, kernel_1.col):
                    # kernel pixel displacement from kernel origin
                    dy = k - kernel_1.origin[0]
                    dx = l - kernel_1.origin[1]
                    p1 += padded_image[i+dy, j+dx] * kernel_1.pixel(k, l)
                    p2 += padded_image[i+dy, j+dx] * kernel_2.pixel(k, l)
            gradient_magnitude = np.linalg.norm([p1, p2])
            if(gradient_magnitude < threshold):
                result[i, j] = 255
    
    return result

def SobelEdgeDetector(image, threshold):
    result = np.zeros(image.shape, np.float32)
    row, col = image.shape
    padded_image = np.array(cv.copyMakeBorder(image, 1, 1, 1, 1, cv.BORDER_REPLICATE), dtype=np.float32)
  
    kernel_1_arr = np.array([   [-1, -2, -1],
                                [0, 0, 0],
                                [1, 2, 1]])
    kernel_2_arr = np.transpose(kernel_1_arr)

    kernel_1 = Kernel(kernel_1_arr, (1, 1))
    kernel_2 = Kernel(kernel_2_arr, (1, 1))

    # # scan each pixel 
    # for i in range(row):
    #     for j in range(col):
    #         p1 = p2 = 0
    #         # run kernel over pixel
    #         for k in range(0, kernel_1.row):
    #             for l in range(0, kernel_1.col):
    #                 # kernel pixel displacement from kernel origin
    #                 dy = k - kernel_1.origin[0]
    #                 dx = l - kernel_1.origin[1]
    #                 p1 += padded_image[i+dy, j+dx] * kernel_1.pixel(k, l)
    #                 p2 += padded_image[i+dy, j+dx] * kernel_2.pixel(k, l)
    #         gradient_magnitude = np.linalg.norm([p1, p2])
    #         if(gradient_magnitude < threshold):
    #             result[i, j] = 255
    GradientEdgeDetect(padded_image, [kernel_1, kernel_2], result, threshold, row, col)
    
    return result

def GradientEdgeDetect(source, kernels, result, threshold, total_row, total_col):
    for i in range(total_row):
        for j in range(total_col):
            p = [] # save kernel convolution values
            k_row = kernels[0].row
            k_col = kernels[1].col
            # convolution 
            for k in range(0, k_row):
                for l in range(0, k_col):
                    # kernel pixel displacement from kernel origin
                    dy = k - kernels[0].origin[0]
                    dx = l - kernels[0].origin[1]
                    # run all kernels
                    for kernel_idx in range(len(kernels)):
                        p[kernel_idx] += source[i+dy, j+dx] * kernels[kernel_idx].pixel(k, l)
            gradient_magnitude = np.linalg.norm(p)
            if(gradient_magnitude < threshold):
                result[i, j] = 255

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source", default="lena.bmp")
    args = parser.parse_args()
    # read image in grayscale mode
    source_image = cv.imread(args.source, 0)

    # Robert
    # robert = RobertsOperator(source_image, 30)
    # cv.imwrite("robert.png", robert)

    # Prewitt
    prewitt = PrewittEdgeDetector(source_image, 24)
    cv.imwrite("prewitt.png", prewitt)

    # Sobel
    sobel = SobelEdgeDetector(source_image, 38)
    cv.imwrite("sobel.png", sobel)
