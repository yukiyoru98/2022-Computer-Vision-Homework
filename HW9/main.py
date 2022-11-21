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

class GradientMagnitudeType:
    SQUARE_SUM_ROOT = 0
    MAX = 1

def GradientEdgeDetect(source, kernels, result, threshold, total_row, total_col, gradientMagnitudeType = GradientMagnitudeType.SQUARE_SUM_ROOT):
    for i in range(total_row):
        for j in range(total_col):
            gradient_magnitude = 0
            conv = np.zeros(len(kernels)) # kernel convolution values
            # pick first kernel to get the kernels' size (all kernels should be the same size)
            k_row = kernels[0].row
            k_col = kernels[1].col
            # kernel convolution 
            for k in range(0, k_row):
                for l in range(0, k_col):
                    # kernel pixel displacement from kernel origin 
                    # origin is determined by the first kernel since all kernels should have the same origin position
                    dy = k - kernels[0].origin[0]
                    dx = l - kernels[0].origin[1]
                    # run all kernels
                    for kernel_idx in range(len(kernels)):
                        conv[kernel_idx] += source[i+dy, j+dx] * kernels[kernel_idx].pixel(k, l)

            # calculate gradient magnitude
            if(gradientMagnitudeType == GradientMagnitudeType.SQUARE_SUM_ROOT):
                gradient_magnitude = np.linalg.norm(conv)
            elif(gradientMagnitudeType == GradientMagnitudeType.MAX):
                gradient_magnitude = np.max(conv)

            if(gradient_magnitude < threshold):
                result[i, j] = 255

def RobertsOperator(image, threshold):
    result = np.zeros(image.shape, np.float32)
    row, col = image.shape
    padded_image = np.array(cv.copyMakeBorder(image, 1, 1, 1, 1, cv.BORDER_REPLICATE), dtype=np.float32)

    kernels = []

    kernel_arr_1 = np.array([   [-1, 0],
                                [ 0, 1]])

    kernel_arr_2 = np.array([   [0, -1],
                                [1, 0]])
    
    kernels.append( Kernel(kernel_arr_1, (0, 0)) )
    kernels.append( Kernel(kernel_arr_2, (0, 0)) )

    GradientEdgeDetect(padded_image, kernels, result, threshold, row, col)

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

    GradientEdgeDetect(padded_image, [kernel_1, kernel_2], result, threshold, row, col)

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

    GradientEdgeDetect(padded_image, [kernel_1, kernel_2], result, threshold, row, col)
    
    return result

def FreiAndChenGradientOperator(image, threshold):
    result = np.zeros(image.shape, np.float32)
    row, col = image.shape
    padded_image = np.array(cv.copyMakeBorder(image, 1, 1, 1, 1, cv.BORDER_REPLICATE), dtype=np.float32)
  
    kernel_1_arr = np.array([   [-1, -math.sqrt(2), -1],
                                [0, 0, 0],
                                [1, math.sqrt(2), 1]])
    kernel_2_arr = np.transpose(kernel_1_arr)

    kernel_1 = Kernel(kernel_1_arr, (1, 1))
    kernel_2 = Kernel(kernel_2_arr, (1, 1))

    GradientEdgeDetect(padded_image, [kernel_1, kernel_2], result, threshold, row, col)
    
    return result


def KirschCompassOperator(image, threshold):
    result = np.zeros(image.shape, np.float32)
    row, col = image.shape
    padded_image = np.array(cv.copyMakeBorder(image, 1, 1, 1, 1, cv.BORDER_REPLICATE), dtype=np.float32)

    kernels = []

    kernel_arr_1 = np.array([   [-3, -3, 5],
                                [-3,  0, 5],
                                [-3, -3, 5]])

    kernel_arr_2 = np.array([   [-3, -3, -3],
                                [-3,  0,  5],
                                [-3,  5,  5]])

    for i in range(4):
        kernels.append( Kernel(kernel_arr_1, (1, 1)) )
        kernels.append( Kernel(kernel_arr_2, (1, 1)) )
        kernel_arr_1 = np.rot90(kernel_arr_1)
        kernel_arr_2 = np.rot90(kernel_arr_2)

    GradientEdgeDetect(padded_image, kernels, result, threshold, row, col, gradientMagnitudeType=GradientMagnitudeType.MAX)
    
    return result


def RobinsonCompassOperator(image, threshold):
    result = np.zeros(image.shape, np.float32)
    row, col = image.shape
    padded_image = np.array(cv.copyMakeBorder(image, 1, 1, 1, 1, cv.BORDER_REPLICATE), dtype=np.float32)

    kernels = []

    kernel_arr_1 = np.array([   [-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]])

    kernel_arr_2 = np.array([   [0,   1,  2],
                                [-1,  0,  1],
                                [-2, -1,  0]])

    for i in range(4):
        kernels.append( Kernel(kernel_arr_1, (1, 1)) )
        kernels.append( Kernel(kernel_arr_2, (1, 1)) )
        kernel_arr_1 = np.rot90(kernel_arr_1)
        kernel_arr_2 = np.rot90(kernel_arr_2)

    GradientEdgeDetect(padded_image, kernels, result, threshold, row, col, gradientMagnitudeType=GradientMagnitudeType.MAX)
    
    return result


def Nevatia_Babu_Operator(image, threshold):
    result = np.zeros(image.shape, np.float32)
    row, col = image.shape
    padded_image = np.array(cv.copyMakeBorder(image, 2, 2, 2, 2, cv.BORDER_REPLICATE), dtype=np.float32)

    kernels = []

    kernel_arr_0 = np.array([   [100, 100, 100, 100, 100],
                                [100, 100, 100, 100, 100],
                                [  0,   0,   0,   0,   0],
                                [-100, -100, -100, -100, -100],
                                [-100, -100, -100, -100, -100]])

    kernel_arr_90 = np.transpose(kernel_arr_0)

    kernel_arr_30 = np.array([  [100, 100, 100, 100, 100],
                                [100, 100, 100,  78, -32],
                                [100,  92,   0, -92, -100],
                                [ 32, -78, -100, -100, -100],
                                [-100, -100, -100, -100, -100]])

    kernel_arr_60 = np.transpose(kernel_arr_30)
    kernel_arr_neg60 = np.fliplr(kernel_arr_60)
    kernel_arr_neg30 = np.fliplr(kernel_arr_30)

    kernels.append( Kernel(kernel_arr_0, (2, 2)) )
    kernels.append( Kernel(kernel_arr_90, (2, 2)) )
    kernels.append( Kernel(kernel_arr_30, (2, 2)) )
    kernels.append( Kernel(kernel_arr_60, (2, 2)) )
    kernels.append( Kernel(kernel_arr_neg30, (2, 2)) )
    kernels.append( Kernel(kernel_arr_neg60, (2, 2)) )


    GradientEdgeDetect(padded_image, kernels, result, threshold, row, col, gradientMagnitudeType=GradientMagnitudeType.MAX)
    
    return result



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source", default="lena.bmp")
    parser.add_argument("-m", "--method", default="Robert")
    parser.add_argument("-t", "--threshold", default=30)
    args = parser.parse_args()
    
    threshold = int(args.threshold)
    
    # read image in grayscale mode
    source_image = cv.imread(args.source, 0)

    if(args.method == "Robert"):
        # Robert
        robert = RobertsOperator(source_image, threshold)
        cv.imwrite("robert.png", robert)

    elif(args.method == "Prewitt"):
        # Prewitt
        prewitt = PrewittEdgeDetector(source_image, threshold)
        cv.imwrite("prewitt.png", prewitt)

    elif(args.method == "Sobel"):
        # Sobel
        sobel = SobelEdgeDetector(source_image, threshold)
        cv.imwrite("sobel.png", sobel)

    elif(args.method == "FreiAndChen"):
        # Frei and Chen
        frei_chen = FreiAndChenGradientOperator(source_image, threshold)
        cv.imwrite("frei_and_chen.png", frei_chen)

    elif(args.method == "Kirsch"):
        # Kirsch
        kirsch = KirschCompassOperator(source_image, threshold)
        cv.imwrite("kirsch.png", kirsch)

    elif(args.method == "Robinson"):
        # Robinson
        robinson = RobinsonCompassOperator(source_image, threshold)
        cv.imwrite("robinson.png", robinson)

    elif(args.method == "Nevatia-Babu"):
        # Nevatia-Babu
        nevatia_babu = Nevatia_Babu_Operator(source_image, threshold)
        cv.imwrite("nevatia_babu.png", nevatia_babu)


    
