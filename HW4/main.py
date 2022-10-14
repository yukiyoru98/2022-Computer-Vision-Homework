import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

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
            if(image[i, j] == 1): # if image pixel is white
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
            if(image[i, j] == 1): # if image pixel is white
                # print(f'\nimg[{i}, {j}]')
                fit = True
                # scan each pixel in kernel
                k = 0
                l = 0
                while(fit and k < kernel.row):
                    while(l < kernel.col):
                        # print(f'kernel [{k}, {l}]')
                        # kernel pixel displacement from kernel origin
                        dy = k - kernel.origin[0]
                        dx = l - kernel.origin[1]
                        # print(f'dy, dx [{dy}, {dx}]')
                        # print(i + dy)
                        # print(j + dx)
                        # print(kernel.pixel(k, l) == 0)
                        if(kernel.pixel(k, l) == 1): 
                            if( (i + dy < 0) or (i + dy >= row) or (j + dx < 0) or (j + dx >= col) ): # if kernel pixel is out of image bound
                                # print('out of bound')
                                fit = False
                                break
                            if(image[i+dy, j+dx] == 0): # if kernel pixel has value but corresponding image pixel does not
                                # print('value 0')
                                fit = False
                                break
                        l += 1
                        # print(f'fit : {fit}')

                    k += 1
                
                if(fit):
                    # preserve origin in result image
                    result[i, j] = 1
    return result

                        
                        

if __name__ == "__main__":
    # read source image
    image = cv.imread("lena.bmp", 0) # read with grayscale mode
    # binarize
    binary_image = binarize(image, threshold=128)

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

    # # binary dilation
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