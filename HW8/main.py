import numpy as np
import matplotlib.pyplot as plt
import argparse
import cv2 as cv
import math
import queue
import statistics
from GrayscaleMorphology import grayscale_closing, grayscale_opening, OctogonalKernal
import os
def SaveNoiseImg_PrintSNR(filename, source_image, noise_image):
    # path 
    try: 
        os.mkdir("Output") 
    except: 
        pass
    # cv.imwrite(filename + ".png", noise_image)
    plt.imshow(noise_image, cmap='gray')
    snr = SNR(source_image, noise_image)
    plt.title(f'[{filename}] SNR:{snr:.3f}')
    path = os.path.join("Output", filename + ".png")
    plt.savefig(path, bbox_inches='tight')
    plt.clf()
    print(f'[{filename}]SNR:{snr}')

def SNR(source_image, noise_image):
    # normalize images
    normalized_source_image = source_image.copy()
    normalized_noise_image = noise_image.copy()
    normalized_source_image = (normalized_source_image / 255).astype(float)
    normalized_noise_image = (normalized_noise_image / 255).astype(float)

    row, col = normalized_source_image.shape
    
    total_pixels = row * col

    sum_source_I = 0
    uN_nominator = 0
    for i in range(row):
        for j in range(col):
            sum_source_I += normalized_source_image[i, j]
            dif = normalized_noise_image[i, j] - normalized_source_image[i, j]
            uN_nominator += dif
            
    u = sum_source_I / total_pixels
    uN = uN_nominator / total_pixels

    VS_nominator = 0
    VN_nominator = 0

    for i in range(row):
        for j in range(col):
            VS_nominator += math.pow((normalized_source_image[i, j] - u), 2)
            VN_nominator += math.pow((normalized_noise_image[i, j] - normalized_source_image[i, j] - uN), 2)
    
    VS = VS_nominator / total_pixels
    VN = VN_nominator / total_pixels
    
    return 20 * math.log10(math.sqrt(VS / VN))

def GaussianNoise(image, amplitude):
    result = np.zeros(image.shape)
    row, col = image.shape

    for i in range(row):
        for j in range(col):
            result[i, j] = min(255, image[i, j] + int(amplitude * np.random.normal(0.0, 1.0)))

    return result

def SaltAndPepperNoise(image, threshold):
    result = image.copy()
    row, col = image.shape

    for i in range(row):
        for j in range(col):
            rv = np.random.uniform(0.0, 1.0)
            if(rv < threshold):
                result[i, j] = 0
            elif(rv > 1 - threshold):
                result[i, j] = 255

    return result

def BoxFilter(image, f_height, f_width): # height and width are odd
    result = np.zeros(image.shape, np.uint8)
    row, col = image.shape

    total_filter_pixels = f_height * f_width

    # image padding
    f_half_width = int(f_width / 2)
    f_half_height = int(f_height / 2)
    padded_image = cv.copyMakeBorder(image, f_half_height, f_half_height, f_half_width, f_half_width, borderType=cv.BORDER_REFLECT)
    
    # for each input pixel, apply filter (separable)
    for i in range(row):
        # new IBUF queue for each row
        ibuf_queue = queue.Queue()

        isum = 0
        for j in range(col):

            if(j == 0): # for first pixel
                # initialize ISUM and IBUF queue
                for l in range(-f_half_width, f_half_width + 1, 1): # for each filter column
                    col_sum = 0
                    for k in range(-f_half_height, f_half_height + 1, 1): # sum up pixel values in this column
                        col_sum += padded_image[i + k, j + l]
                    ibuf_queue.put(col_sum)
                    isum += col_sum
            else:
                # add new column
                col_sum = 0
                for k in range(-f_half_height, f_half_height + 1, 1): # sum  
                    col_sum += padded_image[i + k, j + f_half_width]
                ibuf_queue.put(col_sum)
                # ISUM : subtract old column and add new column
                isum = isum - ibuf_queue.get() + col_sum 
            
            result[i, j] = isum / total_filter_pixels

    return result

def MedianFilter(image, f_height, f_width):
    result = np.zeros(image.shape, np.uint8)
    row, col = image.shape
    # image padding
    f_half_width = int(f_width / 2)
    f_half_height = int(f_height / 2)
    padded_image = cv.copyMakeBorder(image, f_half_height, f_half_height, f_half_width, f_half_width, borderType=cv.BORDER_REFLECT)
    
    # for each input pixel
    for i in range(row):
        for j in range(col):
            pixels = [] # get all pixels covered by filter and store into list
            median = 0
            for k in range(-f_half_height, f_half_height + 1, 1): # scan all pixels covered by filter
                for l in range(-f_half_width, f_half_width + 1, 1): 
                   pixels.append(padded_image[i + k, j + l])
            
            median = statistics.median(pixels)
            result[i, j] = median

    return result



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source", default="lena.bmp")
    args = parser.parse_args()

    # read image in grayscale mode
    source_image = cv.imread(args.source, 0)
    
    # add gaussian noise - amplitude 10
    G_noise_10 = GaussianNoise(source_image, 10)
    # add gaussian noise - amplitude 30
    G_noise_30 = GaussianNoise(source_image, 30)
    # add salt&pepper noise - threshold 0.05
    SP_noise_005 = SaltAndPepperNoise(source_image, 0.05)
    # add salt&pepper noise - threshold 0.1
    SP_noise_01 = SaltAndPepperNoise(source_image, 0.1)

    # save noise images
    SaveNoiseImg_PrintSNR("G_noise_10", source_image, G_noise_10)
    SaveNoiseImg_PrintSNR("G_noise_30", source_image, G_noise_30)
    SaveNoiseImg_PrintSNR("SP_noise_005", source_image, SP_noise_005)
    SaveNoiseImg_PrintSNR("SP_noise_01", source_image, SP_noise_01)

    oct_kernel = OctogonalKernal()

    # ======For Gaussian noise 10======

    # Box filter 3x3
    box_3x3 = BoxFilter(G_noise_10, 3, 3)
    SaveNoiseImg_PrintSNR("box_3x3_G10", source_image, box_3x3)
   
    # Box filter 5x5
    box_5x5 = BoxFilter(G_noise_10, 5, 5)
    SaveNoiseImg_PrintSNR("box_5x5_G10", source_image, box_5x5)
    
    # Median filter 3x3
    median_3x3 = MedianFilter(G_noise_10, 3, 3)
    SaveNoiseImg_PrintSNR("median_3x3_G10", source_image, median_3x3)    
    
    # Median filter 5x5
    median_5x5 = MedianFilter(G_noise_10, 5, 5)
    SaveNoiseImg_PrintSNR("median_5x5_G10", source_image, median_5x5)    
    
    # Opening-then-closing
    open_close = grayscale_closing(grayscale_opening(G_noise_10, oct_kernel), oct_kernel)
    SaveNoiseImg_PrintSNR("open_close_G10", source_image, open_close)    
    
    # Closing-then-opening
    close_open = grayscale_opening(grayscale_closing(G_noise_10, oct_kernel), oct_kernel)
    SaveNoiseImg_PrintSNR("close_open_G10", source_image, close_open)    

    # ======For Gaussian noise 30======

    # Box filter 3x3
    box_3x3 = BoxFilter(G_noise_30, 3, 3)
    SaveNoiseImg_PrintSNR("box_3x3_G30", source_image, box_3x3)
   
    # Box filter 5x5
    box_5x5 = BoxFilter(G_noise_30, 5, 5)
    SaveNoiseImg_PrintSNR("box_5x5_G30", source_image, box_5x5)
    
    # Median filter 3x3
    median_3x3 = MedianFilter(G_noise_30, 3, 3)
    SaveNoiseImg_PrintSNR("median_3x3_G30", source_image, median_3x3)    
    
    # Median filter 5x5
    median_5x5 = MedianFilter(G_noise_30, 5, 5)
    SaveNoiseImg_PrintSNR("median_5x5_G30", source_image, median_5x5)    
    
    # Opening-then-closing
    open_close = grayscale_closing(grayscale_opening(G_noise_30, oct_kernel), oct_kernel)
    SaveNoiseImg_PrintSNR("open_close_G30", source_image, open_close)    
    
    # Closing-then-opening
    close_open = grayscale_opening(grayscale_closing(G_noise_30, oct_kernel), oct_kernel)
    SaveNoiseImg_PrintSNR("close_open_G30", source_image, close_open)    

    # ======For SP noise 0.05======

    # Box filter 3x3
    box_3x3 = BoxFilter(SP_noise_005, 3, 3)
    SaveNoiseImg_PrintSNR("box_3x3_SP005", source_image, box_3x3)
   
    # Box filter 5x5
    box_5x5 = BoxFilter(SP_noise_005, 5, 5)
    SaveNoiseImg_PrintSNR("box_5x5_SP005", source_image, box_5x5)
    
    # Median filter 3x3
    median_3x3 = MedianFilter(SP_noise_005, 3, 3)
    SaveNoiseImg_PrintSNR("median_3x3_SP005", source_image, median_3x3)    
    
    # Median filter 5x5
    median_5x5 = MedianFilter(SP_noise_005, 5, 5)
    SaveNoiseImg_PrintSNR("median_5x5_SP005", source_image, median_5x5)    
    
    # Opening-then-closing
    open_close = grayscale_closing(grayscale_opening(SP_noise_005, oct_kernel), oct_kernel)
    SaveNoiseImg_PrintSNR("open_close_SP005", source_image, open_close)    
    
    # Closing-then-opening
    close_open = grayscale_opening(grayscale_closing(SP_noise_005, oct_kernel), oct_kernel)
    SaveNoiseImg_PrintSNR("close_open_SP005", source_image, close_open)    
    
    # ======For SP noise 0.1======

    # Box filter 3x3
    box_3x3 = BoxFilter(SP_noise_01, 3, 3)
    SaveNoiseImg_PrintSNR("box_3x3_SP01", source_image, box_3x3)
   
    # Box filter 5x5
    box_5x5 = BoxFilter(SP_noise_01, 5, 5)
    SaveNoiseImg_PrintSNR("box_5x5_SP01", source_image, box_5x5)
    
    # Median filter 3x3
    median_3x3 = MedianFilter(SP_noise_01, 3, 3)
    SaveNoiseImg_PrintSNR("median_3x3_SP01", source_image, median_3x3)    
    
    # Median filter 5x5
    median_5x5 = MedianFilter(SP_noise_01, 5, 5)
    SaveNoiseImg_PrintSNR("median_5x5_SP01", source_image, median_5x5)    
    
    # Opening-then-closing
    open_close = grayscale_closing(grayscale_opening(SP_noise_01, oct_kernel), oct_kernel)
    SaveNoiseImg_PrintSNR("open_close_SP01", source_image, open_close)    
    
    # Closing-then-opening
    close_open = grayscale_opening(grayscale_closing(SP_noise_01, oct_kernel), oct_kernel)
    SaveNoiseImg_PrintSNR("close_open_SP01", source_image, close_open)    