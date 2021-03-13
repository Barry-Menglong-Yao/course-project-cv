# from union_find_forest import UnionFindForest
from .union_find_forest import UnionFindForest
from .position import Position
import numpy as np
from tools import data_print
import os
from tools  import encoder
import json

forest = UnionFindForest()

def ccl(img):
    return ocl(img)

def ocl(img):
    
    img_label=img.astype( dtype=np.int16)
    mask_background(img_label)
    first_pass(img_label)
    forest.flatten()
    second_pass(img_label)
    block_list=split_to_block(img_label)
    encoder.save_block_list(block_list,data_print.CCL_DIR)
    return block_list






def process(img,x,y):
    if is_not_background(img[y][x]):
        # print("{0} {1} {2} {3} {4} {5} {6}".format(y,x,img[y][x],a(img,x,y),b(img,x,y), c(img,x,y),d(img,x,y)))
        if is_not_background(b(img,x,y)):
            img[y][x]=forest.find_root(b(img,x,y))
        elif is_not_background(c(img,x,y)):
            if is_not_background(a(img,x,y)):
                img[y][x]=forest.union_tree(a(img,x,y),c(img,x,y))
            elif is_not_background(d(img,x,y)):
                img[y][x]=forest.union_tree(d(img,x,y),c(img,x,y))
            else:
                img[y][x]=forest.find_root(c(img,x,y))
        elif is_not_background(a(img,x,y)):
            img[y][x]=forest.find_root(a(img,x,y))
        elif is_not_background(d(img,x,y)):
            img[y][x]=forest.find_root(d(img,x,y))
        else:
            img[y][x]=forest.new_tree()

        # print("{0}  ".format( img[y][x] ))

def a(img,x,y):
    return img[y-1][x-1]

def b(img,x,y):
    return img[y-1][x]

def c(img,x,y):
    return img[y-1][x+1]

def d(img,x,y):
    return img[y][x-1]

def is_not_background(pixel):
    
    if pixel!=-1:
        return True
    else:   
        return False

def first_pass(img):
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            process(img,x,y)
    # data_print.print_img_to_file(img,data_print.IMG_CC_DIR)

def mask_background(img):
    threshold=175 # TODO
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if img[y][x]>=threshold:
                img[y][x]=-1


def second_pass(img):
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if is_not_background(img[y][x]):
                img[y][x]=forest.find_root(img[y][x])
            else:
                img[y][x]=0
     



def split_to_block(img):
    position_dict=dict()
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if img[y][x] not in position_dict:
                position_dict[img[y][x]]=Position(img[y][x],x,x,y,y)
            else:
                if x>position_dict[img[y][x]].x_right:
                    position_dict[img[y][x]].x_right=x
                elif x<position_dict[img[y][x]].x_left:
                    position_dict[img[y][x]].x_left=x
                if y>position_dict[img[y][x]].y_down:
                    position_dict[img[y][x]].y_down=y
                elif y<position_dict[img[y][x]].y_top:
                    position_dict[img[y][x]].y_top=y


    position_list=position_dict.values()
    sorted( position_list, key=lambda x: (x.x_left,x.y_top), reverse=True)
     

    for position in position_list:
        position.w=position.x_right-position.x_left+1
        position.h=position.y_down-position.y_top+1
        # print("{0}  ".format( img[y][x] ))
    return position_list



def main():
    
    return




if __name__ == "__main__":
    main()