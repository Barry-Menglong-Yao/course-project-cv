# 1. Only add your code inside the function (including newly improted packages). 
#  You can design a new function and call the new function in the given functions. 
# 2. For bonus: Give your own picturs. If you have N pictures, name your pictures such as ["t3_1.png", "t3_2.png", ..., "t3_N.png"], and put them inside the folder "images".
# 3. Not following the project guidelines will result in a 10% reduction in grades

import cv2
import numpy as np
import matplotlib.pyplot as plt
import t1


class Overlap:
    def __init__(self,base_img_idx,compared_img_idx, is_overlap=0, h=None):
        self.is_overlap=is_overlap
        self.h=h   
        self.base_img_idx = base_img_idx
        self.compared_img_idx=compared_img_idx

     

def stitch(imgmark, N=4, savepath=''): #For bonus: change your input(N=*) here as default if the number of your input pictures is not 4.
    "The output image should be saved in the savepath."
    "The intermediate overlap relation should be returned as NxN a one-hot(only contains 0 or 1) array."
    "Do NOT modify the code provided."
    imgpath = [f'./images/{imgmark}_{n}.png' for n in range(1,N+1)]
    imgs = []
    for ipath in imgpath:
        img = cv2.imread(ipath)
        imgs.append(img)
    "Start you code here"
    is_draw=True
    overlap_list=[]
    for i in range(len(imgs)):
        for j in range(i+1,len(imgs)):
            base_img=imgs[i]
            compared_img=imgs[j]
            if base_img is not None:
                overlap=compute_overlap(base_img,compared_img,i,j,is_draw)
                overlap_list.append(overlap)

    overlap_arr=gen_overlap_arr(overlap_list,len(imgs))
    if len(overlap_list)>0:
        stitch_multi_imgs(imgs,overlap_arr,savepath,is_draw)

    return overlap_arr

def stitch_multi_imgs(imgs,overlap_arr,savepath,is_draw):
    first_base_img_idx=find_first_base_img(overlap_arr)
    stitch_order=[first_base_img_idx]
    stitch_order=gen_stitch_order(stitch_order,first_base_img_idx,overlap_arr)

    compared_img=imgs[stitch_order[0]]
    for i in range(1,len(stitch_order)):
        base_img=imgs[stitch_order[i]]
        compared_img=t1.stitch_two_imgs(base_img,compared_img,is_draw,"add_{}.jpg".format(i),True)

    t1.save_img_to_file(compared_img,savepath)

def gen_stitch_order(stitch_order,look_up_img_idx,overlap_arr):
    for i in range(overlap_arr.shape[0]):
        if overlap_arr[look_up_img_idx][i]==1 and i not in stitch_order:
            stitch_order.append(i)
            stitch_order=gen_stitch_order(stitch_order,i,overlap_arr)
    return stitch_order

def find_first_base_img(overlap_arr):
    for i in range(overlap_arr.shape[0]):
        overlap_num=np.sum(overlap_arr[i])
        if overlap_num>1:
            return i    
    return -1


def compute_overlap(base_img,compared_img,i,j,is_draw):
    overlap=Overlap(base_img_idx=i,compared_img_idx=j)
 
    pad_base_img,keypoints1,keypoints2,match_list,h,matchesMask=t1.compute_warp(base_img,compared_img,is_draw)
    overlap.h=h
    if(is_overlapped(h,base_img,compared_img,pad_base_img,is_draw)):
        overlap.is_overlap=1
    return overlap

def is_overlapped(h,base_img,compared_img,pad_base_img,is_draw):
    height,w,d = compared_img.shape
    pts = np.float32([ [0,0],[0,height-1],[w-1,height-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,h)

    iou=t1.find_intersection(h,base_img,compared_img,pad_base_img,is_draw)
    if iou>0.15:
        return True
    else:
        return False
    # for destination_point in dst:
    #     if(is_in(destination_point,base_img,pad_base_img)):
    #         return True
    # return False

def is_in(destination_point,base_img,pad_base_img):
    import t1
    is_in_flag=False
    top=t1.top_distance(base_img)
    down=pad_base_img.shape[0]-t1.top_distance(base_img)
    left=t1.left_distance(base_img)
    right=pad_base_img.shape[1]-t1.left_distance(base_img)
    if(destination_point[0][1]>top and destination_point[0][1]<down and destination_point[0][0]>left and destination_point[0][0]<right):
        is_in_flag=True
         
    
    return is_in_flag

def gen_overlap_arr(overlap_list,number):
    overlap_matrix=np.eye(number)
    for overlap in overlap_list:
        if overlap.is_overlap:
            overlap_matrix[overlap.base_img_idx][overlap.compared_img_idx]=1
            overlap_matrix[overlap.compared_img_idx][overlap.base_img_idx]=1
    return overlap_matrix

if __name__ == "__main__":
    #task2
    import json
    overlap_arr = stitch('t2', N=4, savepath='task2.png')
    with open('t2_overlap.txt', 'w') as outfile:
        json.dump(overlap_arr.tolist(), outfile)
    #bonus
    overlap_arr2 = stitch('t3', savepath='task3.png')
    with open('t3_overlap.txt', 'w') as outfile:
        json.dump(overlap_arr2.tolist(), outfile)
