import numpy as np
import matplotlib.pyplot as plt
import cv2
import math

 
def concat_tile_resize(im_list_2d, interpolation=cv2.INTER_CUBIC):
    im_list_v = [hconcat_resize_min(im_list_h, interpolation=cv2.INTER_CUBIC) for im_list_h in im_list_2d]
    return vconcat_resize_min(im_list_v, interpolation=cv2.INTER_CUBIC)

def vconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    w_min = min(im.shape[1] for im in im_list)
    im_list_resize = [cv2.resize(im, (w_min, int(im.shape[0] * w_min / im.shape[1])), interpolation=interpolation)
                      for im in im_list]
    return cv2.vconcat(im_list_resize)

def hconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    h_min = min(im.shape[0] for im in im_list)
    im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation)
                      for im in im_list]
    return cv2.hconcat(im_list_resize)

def concat_multi_imgs(im_list_2d,output_path):
    im_tile_resize = concat_tile_resize(im_list_2d)
    cv2.imwrite(output_path, im_tile_resize)

def concat_multi_faces_by_cluster(cluster_list,face_list,image_list):
    for cluster in cluster_list:
        concat_multi_faces_of_one_cluster(cluster,face_list,image_list)


def concat_multi_faces_of_one_cluster(cluster,face_list,image_list):
    face_matrix=gen_face_matrix(cluster,face_list,image_list)
    concat_multi_imgs(face_matrix,"cluster_"+str(cluster.cluster_no)+".jpg")

def gen_face_matrix(cluster,face_list,image_list):
    length=len(cluster.element_id_list)
    horizontal_img_num=math.isqrt(length)
    vertical_img_num=length//horizontal_img_num
    if length>vertical_img_num*horizontal_img_num:
        vertical_img_num=vertical_img_num+1
    face_matrix=[]
    for i in range(vertical_img_num):
        horizontal_face_list=[]
        for j in range(horizontal_img_num):
            img_idx=i*horizontal_img_num+j 
            if img_idx<length:
                horizontal_face=crop_face(face_list[cluster.element_id_list[img_idx]],image_list)
                horizontal_face_list.append(horizontal_face)
            else:
                break
        face_matrix.append(horizontal_face_list)
    return face_matrix

def crop_face(face ,image_list):
    its_image=image_list[face.img_idx].rgb_img
    croped_face=im_crop(its_image,face.x,face.y,face.w,face.h)   
    return croped_face 

def im_crop(img,x1,y1,w,h):
    x2=x1+w 
    y2=y1+h
    return img[y1:y2, x1:x2, :]