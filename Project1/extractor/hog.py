import cv2 
# help(cv2.HOGDescriptor)
 
from tools import data_print
 

def hog(img):

    winSize = (20,20)
    blockSize = (10,10)
    blockStride = (5,5)
    cellSize = (10,10)
    nbins = 9
    derivAperture = 1
    winSigma = -1.
    histogramNormType = 0
    L2HysThreshold = 0.2
    gammaCorrection = 1
    nlevels = 64
    signedGradients = True

    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,
        winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels, signedGradients)
    descriptor = hog.compute(img)

    # show(img,descriptor)
    return descriptor


def show(img,descriptor):
    data_print.print_img(img)
    # data_print.show_histogram(descriptor)

    