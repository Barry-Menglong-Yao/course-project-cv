# 1. Only add your code inside the function (including newly improted packages). 
#  You can design a new function and call the new function in the given functions. 
# 2. For bonus: Give your own picturs. If you have N pictures, name your pictures such as ["t3_1.png", "t3_2.png", ..., "t3_N.png"], and put them inside the folder "images".
# 3. Not following the project guidelines will result in a 10% reduction in grades

import cv2
import numpy as np
import matplotlib.pyplot as plt
 



     

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
    is_draw=False
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
    if first_base_img_idx>=0:
        stitch_order=[first_base_img_idx]
        stitch_order=gen_stitch_order(stitch_order,first_base_img_idx,overlap_arr)

        
        compared_img=imgs[stitch_order[0]]
        height,w,_=compared_img.shape
        for i in range(1,len(stitch_order)):
            base_img=imgs[stitch_order[i]]
            compared_img=stitch_two_imgs(base_img,compared_img,is_draw,"add_{}.jpg".format(i),False)
            compared_img=cv2.resize(compared_img, (w*2, height*2)) 

    
        cv2.imwrite(savepath,compared_img) 



class Overlap:
    def __init__(self,base_img_idx,compared_img_idx, is_overlap=0, h=None):
        self.is_overlap=is_overlap
        self.h=h   
        self.base_img_idx = base_img_idx
        self.compared_img_idx=compared_img_idx


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
 
    pad_base_img,keypoints1,keypoints2,match_list,h,matchesMask=compute_warp(base_img,compared_img,is_draw)
    overlap.h=h
    if(is_overlapped(h,base_img,compared_img,pad_base_img,is_draw)):
        overlap.is_overlap=1
    return overlap

def is_overlapped(h,base_img,compared_img,pad_base_img,is_draw):
    # height,w,d = compared_img.shape
    # pts = np.float32([ [0,0],[0,height-1],[w-1,height-1],[w-1,0] ]).reshape(-1,1,2)
    # dst = cv2.perspectiveTransform(pts,h)

    iou=find_intersection(h,base_img,compared_img,pad_base_img,is_draw)
    if iou>0.15:
        return True
    else:
        return False
  

def gen_overlap_arr(overlap_list,number):
    overlap_matrix=np.eye(number)
    for overlap in overlap_list:
        if overlap.is_overlap:
            overlap_matrix[overlap.base_img_idx][overlap.compared_img_idx]=1
            overlap_matrix[overlap.compared_img_idx][overlap.base_img_idx]=1
    return overlap_matrix















## common
 


def match(query_descriptors,train_descriptors):
    from scipy.spatial import distance as dist
   
    match_list=[]
    for i in range(train_descriptors.shape[0]):
        best_match=cv2.DMatch()
        second_best_match=cv2.DMatch()
        
        match_list_i=[]
        for j in range(query_descriptors.shape[0]):
            distance=dist.sqeuclidean(train_descriptors[i],query_descriptors[j]) 
 
            if(distance<best_match.distance):
                second_best_match.distance=best_match.distance
                second_best_match.trainIdx=best_match.trainIdx
                second_best_match.queryIdx=best_match.queryIdx
                second_best_match.imgIdx=best_match.imgIdx
                best_match.distance=distance
                best_match.trainIdx=i
                best_match.queryIdx=j 
                best_match.imgIdx=0
            elif(distance<second_best_match.distance):
                second_best_match.distance=distance
                second_best_match.trainIdx=i
                second_best_match.queryIdx=j 
                second_best_match.imgIdx=0
        match_list_i.append(best_match)
        match_list_i.append(second_best_match)
        match_list.append(match_list_i)
    good_match_list=ratio_test(match_list)
    good_match_list = sorted(good_match_list, key = lambda x:x.distance)
    return good_match_list[:15]


def ratio_test(match_list):
    good_match_list=[]
    for m,n in match_list:
        if m.distance<0.75*n.distance:
            good_match_list.append(m)
    return good_match_list


 



def stitch_two_imgs(img1, img2,is_draw, savepath,is_save):
    pad_img1,keypoints1,keypoints2,match_list,h,matchesMask=compute_warp(img1,img2,is_draw)
    result=combine(pad_img1, img2,keypoints1,keypoints2,match_list,img1,is_draw,matchesMask,h,savepath)
    if is_save:
        cv2.imwrite(savepath,result) 
    return result


def unpad(compared_img,pad_base_img):
    top= top_distance(compared_img)
    down=pad_base_img.shape[0]- top_distance(compared_img)
    left= left_distance(compared_img)
    right=pad_base_img.shape[1]- left_distance(compared_img)
    return top,down,left,right


def pad(base_img,compared_img):
    borderType = cv2.BORDER_CONSTANT
    top = top_distance(compared_img)
    bottom = top
    left = left_distance(compared_img)
    right = left
    value = [ 0, 0, 0]
    dst = cv2.copyMakeBorder(base_img, top, bottom, left, right, borderType, None, value)
    return dst

def top_distance(img):
    return int(0.8 * img.shape[0])   

def left_distance(img):
    return int(0.8 * img.shape[1])   

def combine(pad_img1, img2,keypoints1,keypoints2,match_list,img1,is_draw,matchesMask,h,savepath):
    

    warped_img = cv2.warpPerspective(img2, h, (pad_img1.shape[1]  , pad_img1.shape[0] ))
    # save_img_to_file(warped_img,"data/warped_img.txt")
    # save_img_to_file(img1,"data/img1.txt")
    # cv2.imwrite("data/warped_img_j.jpg",warped_img) #"data/result.jpg"
    # cv2.imwrite("data/img1_j.jpg",img1) #

    # if is_draw:
    #     draw_img(warped_img)


    top,down,left,right=unpad(img2,pad_img1)
     
    warped_img[top:down, left:right ] = pad_img1[top:down, left:right ]
    
    if is_draw:
        warp_img2_in_img1=draw_warp_img2_in_img1(pad_img1,img2,keypoints1,keypoints2,match_list,matchesMask,h)
        draw_img(warped_img)
 

    return warped_img


def compute_warp(img1,img2,is_draw):#base_img,compared_img
    pad_img1=pad(img1,img2)
    sift =cv2.xfeatures2d.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(pad_img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

 

    # match_list= match(descriptors2 ,descriptors1 ) #   remove comment

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors2,descriptors1,k=2)
    match_list=ratio_test(matches)
    match_list = sorted(match_list, key = lambda x:x.distance)
    match_list=match_list[:15]

    # if is_draw:
    #     draw_match(img2,keypoints2,pad_img1,keypoints1,match_list)

    destination_pts=np.float32([keypoints1[m.trainIdx].pt for m in match_list]).reshape(-1,1,2)
    source_pts=np.float32([keypoints2[m.queryIdx].pt for m in match_list]).reshape(-1,1,2)
    h , mask = cv2.findHomography(source_pts, destination_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()
    return pad_img1,keypoints1,keypoints2,match_list,h,matchesMask
 

def draw_warp_img2_in_img1(img1,img2,keypoints1,keypoints2,match_list,matchesMask,h):
    height,w,d = img2.shape
    pts = np.float32([ [0,0],[0,height-1],[w-1,height-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,h)
    img1 = cv2.polylines(img1,[np.int32(dst)],True,255,3, cv2.LINE_AA)
    return draw_inliers(img1,keypoints1,img2,keypoints2,match_list,matchesMask)

def draw_img(img):
    cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
    resized_img=cv2.resize(img, (1500, 750))      
    cv2.imshow('image', resized_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def draw_inliers(img1,kp1,img2,kp2,match,matchesMask):
    draw_params = dict(matchColor = (0,255,0),  
                   singlePointColor = None,
                   matchesMask = matchesMask,  
                   flags = 2)
    img3 = cv2.drawMatches(img2,kp2,img1,kp1,match,None,**draw_params)
    # plt.imshow(img3, 'gray'),plt.show()
    draw_img(img3)
    return img3
    
    
## common




def find_intersection(h,base_img,compared_img,pad_base_img,is_draw):
    from shapely.geometry import Polygon
    height,w,d = compared_img.shape
    pts = np.float32([ [0,0],[0,height-1],[w-1,height-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,h)

    top,down,left,right=unpad(compared_img,pad_base_img)
    base_box = [[left, top], [left, down], [right, down], [right, top]]
    compared_box = dst.reshape(4,-1).tolist()
    # img1 = cv2.polylines(pad_base_img,[np.int32(dst)],True,255,3, cv2.LINE_AA)
    # img1 = cv2.polylines(img1,[np.int32(base_box)],True,0,3, cv2.LINE_AA)
    # if is_draw:
    #     draw_img(img1)


    poly_1 = Polygon(base_box)
    poly_2 = Polygon(compared_box)
    if(poly_2.area>1):
        intersection=poly_1.intersection(poly_2)
        iou =  intersection.area/ poly_1.union(poly_2).area

        # img1 = cv2.polylines(pad_base_img,[np.int32(intersection.exterior.coords)],True,255,3, cv2.LINE_AA)
        # if is_draw:
        #     draw_img(img1)
        return iou
    else:
        return 0






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
