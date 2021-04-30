#Only add your code inside the function (including newly improted packages)
# You can design a new function and call the new function in the given functions. 
# Not following the project guidelines will result in a 10% reduction in grades

import cv2
import numpy as np
import matplotlib.pyplot as plt
 


def main():
    img1 = cv2.imread('./images/t1_1.png')
    img2 = cv2.imread('./images/t1_2.png')
    savepath = 'task1.png'
    stitch_background(img1, img2, savepath=savepath)



def stitch_background(img1, img2, savepath=''):
    "The output image should be saved in the savepath."
    "Do NOT modify the code provided."
    
    # draw_img(img1)
    # draw_img(img2)

    is_draw=False
    stitch_two_imgs(img1, img2, is_draw,savepath,True)
    
    return np


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

 

    match_list= match(descriptors2 ,descriptors1 ) #   remove comment

    # bf = cv2.BFMatcher()
    # matches = bf.knnMatch(descriptors2,descriptors1,k=2)
    # match_list=ratio_test(matches)
    # match_list = sorted(match_list, key = lambda x:x.distance)
    # match_list=match_list[:15]

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

def test():
    img1 = cv2.imread('./images/t2_2.png')
    img2 = cv2.imread('./images/t2_1.png')
    savepath = 'data/t2/test13.png'
    is_draw=True
    stitch_two_imgs(img1, img2, is_draw,savepath,True)

if __name__ == "__main__":
    # test()
    main()