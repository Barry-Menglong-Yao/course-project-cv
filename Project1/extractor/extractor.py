
from .intensity import sampledIntensityHistogram
from .hog import hog
import cv2 
from tools import data_print

def extract_feature(img):
    dim=(20,20)
    # data_print.print_img(img)
    resized_img=cv2.resize(img,dim)
    # data_print.print_img(resized_img)
    # return sampledIntensityHistogram(img)
    return hog(resized_img)