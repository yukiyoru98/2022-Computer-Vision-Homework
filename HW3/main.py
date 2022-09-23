import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import argparse

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

def getHistogram(image):
    row, col = image.shape
    histogram = np.zeros(256)

    for i in range(row):
        for j in range(col):
            intensity = image[i, j]
            histogram[intensity] += 1
    return histogram

def drawHistrogram(filename, histogram):
    plt.bar(np.arange(256), histogram, width=1.0,color='black')
    plt.xlabel("Intensity")
    plt.ylabel("Pixels")
    plt.title("Histogram")
    plt.savefig(filename)
    plt.clf()
    return

def adjustIntensity(image, multiplier): # input image should be grayscale
    row, col = image.shape
    result = np.zeros(image.shape, np.uint8)

    for i in range(row):
        for j in range(col):
            result[i, j] = image[i, j] * multiplier

    return result

def histogramEquilize(image, histogram): # input image should be grayscale
    row, col = image.shape
    result = np.zeros(image.shape, np.uint8)
    intensity_map = np.zeros(histogram.shape)
    
    # calculate cdf and normalize to 0-255 to get the new intensity value
    total_pixels =  row * col
    sum = 0
    cdf_min = -1
    for i in range(256):
        sum += histogram[i]
        cdf = sum / total_pixels
        if(cdf_min < 0 and histogram[i] > 0):
            cdf_min = cdf
        intensity_map[i] = round(255 * (cdf - cdf_min))
    
    # recolor the original image with the equalized intensity value
    for i in range(row):
        for j in range(col):
            original_intensity = image[i, j]
            result[i, j] = intensity_map[original_intensity]
    return result



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source",   default="lena.bmp")
    parser.add_argument("-m", "--intensity_multiplier",   default=1/3)
    args = parser.parse_args()

    # read source image
    image = cv.imread(args.source) 
    grayscale_image = cv.cvtColor(image, cv.COLOR_RGB2GRAY) # convert to grayscale mode
    # save image
    cv.imwrite('original_image.bmp', image)
    # draw original histogram
    original_histogram = getHistogram(grayscale_image)
    drawHistrogram('original_histogram.png', original_histogram)

    # intensity divided by 3 
    dark_image = adjustIntensity(grayscale_image, float(args.intensity_multiplier) )
    # save image
    cv.imwrite('dark_image.bmp', dark_image)
    # draw histogram
    dark_histogram = getHistogram(dark_image)
    drawHistrogram('dark_histogram.png', dark_histogram)

    # histogram equalization
    eq_image = histogramEquilize(dark_image, dark_histogram)
    # save image
    cv.imwrite('equalized_image.bmp', eq_image)
    # draw histogram
    equalized_histogram = getHistogram(eq_image)
    drawHistrogram('equalized_histogram.png', equalized_histogram)
    
    