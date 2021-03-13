# from matplotlib import pyplot as plt  # TODO delete
import cv2

DATA_DIR="./"
JPG_DIR=DATA_DIR 
CHARACTER_DIR=JPG_DIR+"characters"


INTERMEDIATE_DIR="./data/train_data/"
INTENSITY_DIR=INTERMEDIATE_DIR+"intensity/"
CCL_DIR=INTERMEDIATE_DIR+"ccl/"
IMG_CC_DIR=CCL_DIR+"draft.txt"

OUTPUT_DIR="."
FEATURE_DIR=OUTPUT_DIR+"/feature/"


 
     
 
def print_img(img):
    m,n = img.shape
    for i in range(m):
        for j in range(n) :
            text = str(img[i][j])+ " "
            print("{0:^4}".format(text ), end='')
            # print("{0} ".format(), end='')
        print()

def print_img_to_file(img,path):
    with open(path,"w") as out_file:
        m,n = img.shape
        for i in range(m):
            for j in range(n) :
                text = str(img[i][j])+ " "
                print("{0:^4}".format(text ), end='', file=out_file)
                # print("{0} ".format(), end='')
            print(file=out_file)


def show_histogram(histogram):
    # plt.figure()
    # plt.title("Grayscale Histogram")
    # plt.xlabel("grayscale value")
    # plt.ylabel("pixels")
    # # plt.xlim([0.0, 1.0])   

    # plt.plot(   histogram)  
    # plt.show()
    return

def show_image(img, delay=1000):
    """Shows an image.
    """
    # cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
    # cv2.imshow('image', img)
    # cv2.waitKey(delay)
    # cv2.destroyAllWindows()
    return