import argparse
import cv2 as cv
from imageTransform import ImageTransform

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source"     , default = "lena.bmp")
    args = parser.parse_args()

    # read image
    image = cv.imread(args.source)

    # upside-down
    upside_down = ImageTransform.flip_vertically(image)

    # left-right
    left_right = ImageTransform.flip_horizontally(image)

    # diagonal
    diagonal = ImageTransform.flip_diagonally(image)

    cv.imwrite('upside_down.bmp', upside_down)
    cv.imwrite('left_right.bmp', left_right)
    cv.imwrite('diagonal.bmp', diagonal)




