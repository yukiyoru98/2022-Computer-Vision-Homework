import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def saveLabelMap(data):
    cmap = plt.get_cmap('gist_ncar').copy()
    cmap.set_under('white')
    plt.imsave('label_map.png', data, cmap=cmap, vmin=0.1)
    plt.clf()


def binarize(image, threshold): # image should be gray scale
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

def drawHistogram(image):
    row, col = image.shape
    histogram = np.zeros(256)

    for i in range(row):
        for j in range(col):
            intensity = image[i, j]
            histogram[intensity] += 1
    return histogram

class LabelClass:
    def __init__(self, value, y = 0, x = 0):
        self.eqLabel = value
        self.pixelCnt = 0 
        self.rectTop = y
        self.rectBottom = y
        self.rectLeft = x
        self.rectRight = x
        self.weightX = 0
        self.weightY = 0
        return
    
    def SetEquivalence(self, targetClass):
        self.eqLabel = targetClass.eqLabel
        return
    
    def UpdateWeight(self, y, x):
        self.weightX += x
        self.weightY += y
    
    def print(self):
        print(f"{self.eqLabel} : {self.pixelCnt} | ({self.rectLeft},{self.rectTop}), | ({self.rectRight},{self.rectBottom})")


class ConnectComponent:
    def __init__(self, binary_image):
        self.image = binary_image
        self.row = binary_image.shape[0]
        self.col = binary_image.shape[1]
        self.label_map = np.zeros(binary_image.shape, np.uint64) # initialize label image
        self.labels = {} # dictionary for labels { (int)labelValue : (Label)label}
        return
        
    def confirmPixelLabel(self, y, x, pixelLabel): # when pixel is processed in bottom-up last scan, update label info
        if(not (pixelLabel in self.labels)):
            self.labels[pixelLabel] = LabelClass(pixelLabel, y, x)

        label = self.labels[pixelLabel]
        # increment label pixel count
        label.pixelCnt += 1

        # update weight
        label.UpdateWeight(y, x)

        # update label rect boundaries
        if(y < label.rectTop):  label.rectTop = y
        if(y > label.rectBottom):  label.rectBottom = y
        if(x < label.rectLeft):  label.rectLeft = x
        if(x > label.rectRight):  label.rectRight = x

    def getNeighborLabel(self, y, x):
        neighbors = []
        # above
        if(y - 1 >= 0 and self.label_map[y-1, x] != 0):   neighbors.append(self.label_map[y-1, x])
        # below
        if(y + 1 < self.row and self.label_map[y+1, x] != 0):   neighbors.append(self.label_map[y+1, x])
        # left
        if(x - 1 >= 0 and self.label_map[y, x-1] != 0):   neighbors.append(self.label_map[y, x-1])
        # right
        if(x + 1 < self.col and self.label_map[y, x+1] != 0):   neighbors.append(self.label_map[y, x+1])

        return neighbors

    def getConnectedComponents(self):
        row = self.row
        col = self.col
        next_label = 1
        # top-down
        for i in range(row):
            # create local equivalence label class table for current row
            eqTable = {} # dictionary of label classes
            # first scan
            for j in range(col): # process each pixel
                if(self.image[i, j] == 1):
                    neighbors = self.getNeighborLabel(i, j)
                    current_label = 0
                    if(not neighbors):   # no labeled neighbors
                        # new label
                        current_label = next_label
                        next_label += 1
                    else:
                        # follow neighbor label
                        current_label = min(neighbors)

                    # set label for current pixel
                    self.label_map[i, j] = current_label

                    if(not (current_label in eqTable)):    
                        eqTable[current_label] = LabelClass(current_label)
                    
                    # update equivalence class
                    for n in neighbors:
                        if(n != current_label):  
                            if(not (n in eqTable)): # add neighbor label into table if not recorded
                                eqTable[n] = LabelClass(n)
                            eqTable[n].SetEquivalence(eqTable[current_label])

            # second scan
            for j in range(col):
                # relabel pixel label with equivalence label
                current_label = self.label_map[i, j]
                if(current_label != 0):
                    self.label_map[i, j] = eqTable[current_label].eqLabel

        # bottom-up
        for i in range(row - 1, -1, -1):
            # create local equivalence label class table for current row
            eqTable = {} # dictionary of label classes
            #first scan: update equivalent label classes
            for j in range(col - 1, -1, -1): #process each pixel
                current_label = self.label_map[i, j]
                if(current_label != 0): # if pixel is labeled

                    neighbors = self.getNeighborLabel(i, j)

                    if(neighbors): 
                        # change label into min between neighbors and current
                        current_label = min(current_label, min(neighbors))
                        self.label_map[i, j] = current_label

                    if(not (current_label in eqTable)): # add label into table if not recorded
                        eqTable[current_label] = LabelClass(current_label)

                    for n in neighbors:
                        if(n != current_label):  
                            if(not (n in eqTable)): # add neighbor label into table if not recorded
                                eqTable[n] = LabelClass(n)
                            eqTable[n].SetEquivalence(eqTable[current_label])


            # second scan
            for j in range(col - 1, -1, -1):
                # relabel pixel label with equivalence label
                current_label = self.label_map[i, j]
                if(current_label != 0):
                    self.label_map[i, j] = eqTable[current_label].eqLabel
                    self.confirmPixelLabel(i, j, current_label)
            
            saveLabelMap(self.label_map)

        return

    def drawComponents(self, imageToDraw, threshold = 500):
        for label in self.labels.values():
            if(label.pixelCnt >= threshold):
                cv.rectangle(imageToDraw, (label.rectLeft, label.rectTop), (label.rectRight, label.rectBottom), (255, 0, 0))
                centerX = round(label.weightX / label.pixelCnt)
                centerY = round(label.weightY / label.pixelCnt)
                cv.circle(imageToDraw,(centerX, centerY), 3, (0, 0, 255), -1)

        cv.waitKey(0)
        cv.imwrite('labeled_image.png', imageToDraw)
        return

if __name__ == "__main__":
    
    # read source image
    image = cv.imread("lena.bmp", 0) # read with grayscale mode

    # binarize with threshold 128
    binary_img = binarize(image, 128)
    plt.imsave('binary.bmp', binary_img, cmap='gray')

    # draw histogram
    histogram = drawHistogram(image)
    plt.bar(np.arange(256), histogram, width=1.0,color='black')
    plt.xlabel("Intensity")
    plt.ylabel("Pixels")
    plt.title("Histogram")
    plt.savefig('histogram.png')
    
    # connected components
    cc = ConnectComponent(binary_image=binary_img)
    cc.getConnectedComponents()
    cc.drawComponents(cv.cvtColor(image, cv.COLOR_GRAY2BGR))

