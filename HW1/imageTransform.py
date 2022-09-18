import numpy as np

class ImageTransform:

    def flip_vertically(image):

        row, col, channel = image.shape
        result = np.zeros(image.shape, np.uint8)

        # reverse the rows
        for i in range(row):
            reverse_row = row - i - 1
            for j in range(col):
                for c in range(channel):
                    result[i, j, c] = image[reverse_row, j, c]
        return result

    def flip_horizontally(image):

        row, col, channel = image.shape
        result = np.zeros(image.shape, np.uint8)

        # reverse the columns
        for j in range(col):
            reverse_col = col - j - 1
            for i in range(row):
                for c in range(channel):
                    result[i, j, c] = image[i, reverse_col, c]
        return result

    def flip_diagonally(image):

        row, col, channel = image.shape
        result = np.zeros(image.shape, np.uint8)

        # swap rows and columns
        for i in range(row):
            for j in range(col):
                for c in range(channel):
                    result[i, j, c] = image[j, i, c]
        return result
    