"""
Character Detection

The goal of this task is to implement an optical character recognition system consisting of Enrollment, Detection and Recognition sub tasks

Please complete all the functions that are labelled with '# TODO'. When implementing the functions,
comment the lines 'raise NotImplementedError' instead of deleting them.

Do NOT modify the code provided.
Please follow the guidelines mentioned in the project1.pdf
Do NOT import any library (function, module, etc.).
"""


import argparse
import json
import os
import glob
import cv2
import numpy as np

from ccl.ccl import ccl
from tools import data_print 
 
from tools import encoder
from extractor import extractor



#an array of matrices
def read_image(img_path, show=False):
    """Reads an image into memory as a grayscale array.
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    if show:
        data_print.show_image(img)

    return img



def parse_args():
    parser = argparse.ArgumentParser(description="cse 473/573 project 1.")
    parser.add_argument(
        "--test_img", type=str, default=data_print.JPG_DIR+"test_img.jpg",
        help="path to the image used for character detection (do not change this arg)")
    parser.add_argument(
        "--character_folder_path", type=str, default=data_print.CHARACTER_DIR,
        help="path to the characters folder")
    parser.add_argument(
        "--result_saving_directory", dest="rs_directory", type=str, default="./",
        help="directory to which results are saved (do not change this arg)")
    args = parser.parse_args()
    return args

def ocr(test_img, characters):
    """Step 1 : Enroll a set of characters. Also, you may store features in an intermediate file.
       Step 2 : Use connected component labeling to detect various characters in an test_img.
       Step 3 : Taking each of the character detected from previous step,
         and your features for each of the enrolled characters, you are required to a recognition or matching.

    Args:
        test_img : image that contains character to be detected.
        characters_list: list of characters along with name for each character.

    Returns:
    a nested list, where each element is a dictionary with {"bbox" : (x(int), y (int), w (int), h (int)), "name" : (string)},
        x: row that the character appears (starts from 0).
        y: column that the character appears (starts from 0).
        w: width of the detected character.
        h: height of the detected character.
        name: name of character provided or "UNKNOWN".
        Note : the order of detected characters should follow english text reading pattern, i.e.,
            list should start from top left, then move from left to right. After finishing the first line, go to the next line and continue.
        
    """
    #   Add your code here. Do not modify the return and input arguments

    feature_list=enrollment(characters)
    

    block_list=detection(test_img)
    
    recognition(block_list,test_img,feature_list)

    # raise NotImplementedError

# enroll an array of matrices in your system by extracting appropriate features
def enrollment(characters):
    """ Args:
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.
    """
    #   Step 1 : Your Enrollment code should go here.
    feature_list=[]
    for name, character in  characters :
        feature=extractor.extract_feature(character)    
        feature_list.append([name,feature])
    save_feature_list(feature_list)
    # show_feature_list(feature_list)
    return feature_list 
 

def save_feature_list(feature_list):
    for name,feature in  feature_list :
        with open(os.path.join(data_print.FEATURE_DIR, name+'.json'), "w") as file:
            json.dump({"feature":feature,"character":name}, file, cls=encoder.OcrEncoder)

 

    # raise NotImplementedError

def detection(test_img):
    """ 
    Use connected component labeling to detect various characters in an test_img.
    Args:criteria 
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.
    """
    block_list=ccl(test_img)
        # data_print.print_img(img_position)
    # for position in  block_list :
    #     if(position.label!=0):
    #         print("UN: {0}".format( position) )  
    return block_list

    #  Step 2 : Your Detection code should go here.
    # raise NotImplementedError

def recognition(block_list,total_test_img,feature_list):
    """ 
    Args:
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.
    """
    #   Step 3 : Your Recognition code should go here.
    
    for position in  block_list :
        if(position.label!=0):
            recognizing_character=gen_img(total_test_img,position)
            data_print.show_image(recognizing_character)
            img_feature=extractor.extract_feature(recognizing_character)
            position.character=match(img_feature,feature_list)
            print("find character: {1}  in position {0}  ".format(position,position.character))

    encoder.save_block_list(block_list,data_print.OUTPUT_DIR)

    # raise NotImplementedError

def gen_img(test_img,position):
    return test_img[position.y_top:position.y_down,position.x_left:position.x_right]
    # recognizing_character = np.zeros((position.h,position.w), dtype=int)
    # for y in range(position.y_top,position.y_down+1):
    #     for x in range(position.x_left,position.x_right+1):
    #         recognizing_character[y-position.y_top][x-position.x_left] =test_img[y][x]
    # return recognizing_character


def intersection(hist_1, hist_2):
    minima = np.minimum(hist_1, hist_2)
    intersection = np.true_divide(np.sum(minima), np.sum(hist_2))
    return intersection

def l2(hist_1,hist_2):
    l2 =  np.sum(np.power((hist_1-hist_2),2)) 
    return l2

def match(img_feature,feature_list):
    min_intersection=3
    character="UNKNOWN"
    for feature_item in feature_list:
        cur_intersection=l2(feature_item[1],img_feature)
        if cur_intersection<min_intersection:
            min_intersection=cur_intersection
            character=feature_item[0]
        print(" {0} with intersection= {1}  ".format(feature_item[0],cur_intersection))
    return character

def save_results(coordinates, rs_directory):
    """
    Donot modify this code
    """
    results = []
    with open(os.path.join(rs_directory, 'results.json'), "w") as file:
        json.dump(results, file)
 
    

def main():
    args = parse_args()
    run(args.character_folder_path,args.test_img)
    

def run(character_folder_path,test_img_path):
    characters = []

    all_character_imgs = glob.glob(character_folder_path+ "/*")
    
    for each_character in all_character_imgs :
        character_name = "{}".format(os.path.split(each_character)[-1].split('.')[0])
        characters.append([character_name, read_image(each_character, show=False)])

    test_img = read_image(test_img_path)

    results = ocr(test_img, characters)

    # save_results(results, args.rs_directory)

def test_enroll():
    return 


 
def show_feature_list(feature_map):
    for name,feature in  feature_map :
        print("{0}: {1}".format(name,feature))
        # show_histogram(feature)





def show_data():
    all_character_imgs = glob.glob(data_print.CHARACTER_DIR+ "*")
    
    for each_character in all_character_imgs :
        character_name = "{}".format(os.path.split(each_character)[-1].split('.')[0])
        img = read_image(each_character,False)
        data_print.print_img_to_file(img,data_print.INTENSITY_DIR+character_name+".txt")
        # print_img(img )
        # print()

    img = read_image(data_print.JPG_DIR+"test_img.jpg",False)
    data_print.print_img_to_file(img,data_print.INTENSITY_DIR+"test_img.txt")
 

def test_detect():
    characters = []

    all_character_imgs = glob.glob(data_print.CHARACTER_DIR+ "/*")
    
    for each_character in all_character_imgs :
        character_name = "{}".format(os.path.split(each_character)[-1].split('.')[0])
        img=read_image(each_character, show=False)
        characters.append([character_name,img ])
        img_position=ccl(img)
        # data_print.print_img(img_position)
        for position in  img_position :
            if(position.label!=0):
                print("{0} : {1}".format(character_name,position) )   

    img = read_image(data_print.JPG_DIR+"test_img.jpg",False)
    img_position=ccl(img)
        # data_print.print_img(img_position)
    for position in  img_position :
        if(position.label!=0):
            print("UN: {0}".format( position) )  

if __name__ == "__main__":
    main()
    # show_data()
    # test_enroll()
    # test_detect()
