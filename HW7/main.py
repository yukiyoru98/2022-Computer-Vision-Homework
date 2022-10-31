from fileinput import filename
import numpy as np
import argparse
import cv2 as cv
import matplotlib.pyplot as plt

def downsample(image, sample_size):
    row, col = image.shape
    new_row = int(row / sample_size)
    new_col = int(col / sample_size)
    result = np.zeros((new_row, new_col), np.uint8)

    for i in range(new_row):
        for j in range(new_col):
            # for each pixel in the downsampled image,
            # pick the top-left pixel value from the sample block in the original image
            result[i, j] = image[i * sample_size, j * sample_size]

    return result
    

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

class Yokoi_4_connectivity_Operator:

    @staticmethod
    def Compute(image):
        # input should be binary image
        row, col = image.shape
        result = np.zeros(image.shape, np.uint8)
        
        for i in range(row):
            for j in range(col):
            # for each white pixel, compute function f
                if(image[i, j] == 1):
                    result[i, j] = Yokoi_4_connectivity_Operator.function_f(image, i, j)

        return result

    @staticmethod
    def function_f(image, i, j):
        row, col = image.shape
        x0 = image[i, j]

        # get neighbors value
        x1 = x2 = x3 = x4 = x5 = x6 = x7 = x8 = 0

        if(j+1 < col):
            x1 = image[i, j+1]

        if(i-1 >= 0):
            x2 = image[i-1, j]
            if(j+1 < col):
                x6 = image[i-1, j+1]
            if(j-1 >= 0):
                x7 = image[i-1, j-1]

        if(j-1 >= 0):
            x3 = image[i, j-1]

        if(i+1 < row):
            x4 = image[i+1, j]
            if(j+1 < col):
                x5 = image[i+1, j+1]
            if(j-1 >= 0):
                x8 = image[i+1, j-1]

        h_results = {'q' : 0, 'r' : 0, 's' : 0} # count of q/r/s for the 4 corner neighborhoods
        h_results[ Yokoi_4_connectivity_Operator.function_h(x0, x1, x6, x2) ] += 1 # a1
        h_results[ Yokoi_4_connectivity_Operator.function_h(x0, x2, x7, x3) ] += 1 # a2
        h_results[ Yokoi_4_connectivity_Operator.function_h(x0, x3, x8, x4) ] += 1 # a3
        h_results[ Yokoi_4_connectivity_Operator.function_h(x0, x4, x5, x1) ] += 1 # a4

        if(h_results['r'] == 4):    return 5
        return h_results['q']

    @staticmethod
    def function_h(b, c, d, e):
        if(b != c): return 's'
        if(b == c == d == e):   return 'r'
        return 'q'


class Pair_Relationship_Operator:
    @staticmethod
    def Compute(image, mask_value):
        row, col = image.shape
        result = np.zeros(image.shape)
        
        for i in range(row):
            for j in range(col):
                # compute output for each pixel
                result[i, j] = Pair_Relationship_Operator.pixel_output(image, i, j, mask_value)

        return result

    @staticmethod
    def pixel_output(image, i, j, m):
        row, col = image.shape
        x0 = image[i, j]
        # get neighbor values
        x1 = x2 = x3 = x4 = 0
        if(j+1 < col):
            x1 = image[i, j+1]
        if(i-1 >= 0):
            x2 = image[i-1, j]
        if(j-1 >= 0):
            x3 = image[i, j-1]
        if(i+1 < row):
            x4 = image[i+1, j]

        h_sum = 0 # calculate sum of h function's result of the neighbors
        h_sum += Pair_Relationship_Operator.function_h(x1, m)
        h_sum += Pair_Relationship_Operator.function_h(x2, m)
        h_sum += Pair_Relationship_Operator.function_h(x3, m)
        h_sum += Pair_Relationship_Operator.function_h(x4, m)

        if(h_sum >= 1 and x0 == m):    return 'p'
        return 'q'
    
    @staticmethod
    def function_h(a, m):
        if(a == m): return 1
        return 0
    

class Connected_Shrink_Operator:

    @staticmethod
    def function_f(image, i, j):
        row, col = image.shape
        x0 = image[i, j]

        # get neighbors value
        x1 = x2 = x3 = x4 = x5 = x6 = x7 = x8 = 0

        if(j+1 < col):
            x1 = image[i, j+1]

        if(i-1 >= 0):
            x2 = image[i-1, j]
            if(j+1 < col):
                x6 = image[i-1, j+1]
            if(j-1 >= 0):
                x7 = image[i-1, j-1]

        if(j-1 >= 0):
            x3 = image[i, j-1]
            
        if(i+1 < row):
            x4 = image[i+1, j]
            if(j+1 < col):
                x5 = image[i+1, j+1]
            if(j-1 >= 0):
                x8 = image[i+1, j-1]

        h_results = {'1' : 0, '0' : 0} # count of q/r/s for the 4 corner neighborhoods
        h_results[ Connected_Shrink_Operator.function_h(x0, x1, x6, x2) ] += 1 # a1
        h_results[ Connected_Shrink_Operator.function_h(x0, x2, x7, x3) ] += 1 # a2
        h_results[ Connected_Shrink_Operator.function_h(x0, x3, x8, x4) ] += 1 # a3
        h_results[ Connected_Shrink_Operator.function_h(x0, x4, x5, x1) ] += 1 # a4

        if(h_results['1'] == 1):    return True # removable
        return False 

    @staticmethod
    def function_h(b, c, d, e):
        if(b == c and (b != d or b != e)):  return '1'
        return '0'

class Thinning_Operator:
    
    @staticmethod
    def Compute(image):
        row, col = image.shape
        result = image.copy()

        while(True): # iterate until the result doesn't change
            # compute Yokoi connectivity number for whole image
            yokoi = Yokoi_4_connectivity_Operator.Compute(result)

            changed = False

            # for each pixel, compute pair relationship and connected shrink
            for i in range(row):
                for j in range(col):
                    pairRelationship = Pair_Relationship_Operator.pixel_output(yokoi, i, j, m=1) # use Yokoi image as the mask value for pair relationship
                    removable = Connected_Shrink_Operator.function_f(result, i, j)
                    # remove pixel if both conditions are satisfied
                    if(pairRelationship == 'p' and removable):
                        result[i, j] = 0
                        changed = True

            if(changed == False):
                break

        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', "--source", default="lena.bmp")
    args = parser.parse_args()

    # read source image in grayscale mode
    image = cv.imread(args.source, 0)
    # binarize image with threshold=128
    binary_image = binarize(image, threshold=128)
    # downsample to 64x64
    downsampled_image = downsample(binary_image, sample_size=8)
    
    # Apply Thinning operator
    result = Thinning_Operator.Compute(downsampled_image)
    plt.imsave('thinning.bmp', result, cmap='gray')
