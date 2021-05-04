import cv2
import os
import json 
from utils.util import Face
from utils.util import Encoder
from computeFBeta.ComputeFBeta import *
from utils.config import * 
import argparse
class Image:
    def __init__(self, rgb_img,gray_img,name):
        self.rgb_img=rgb_img
        self.pixels=gray_img
        self.name=name 


def face_detect(img_dir,output_path,is_save_json,is_draw):
    face_list=[]
    image_list=read_image_list(img_dir)
    face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    for idx in range(len(image_list)):
        image=image_list[idx]
        face_position_list=face_cascade.detectMultiScale(image.pixels,scaleFactor=1.2, minNeighbors=5, minSize=(5, 5))   #
        for face_position in face_position_list:
            x,y,w,h=face_position 
            face=Face(idx,image.name,x,y,w,h)
            face_list.append(face)
        if is_draw:
            draw_face(image.pixels,face_position_list)
    if is_save_json:
        with open(output_path, 'w') as f:
            json.dump(face_list, f, cls=Encoder)
    return face_list,image_list
 

def draw_face(img,faces):
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    # Display the output
    cv2.imshow('img', img)
    cv2.waitKey()

def gen_Face(idx,face,img_name):
    x,y,w,h=face 
    Face(idx,img_name,x,y,w,h)
    # json_object = {"iname": img_name, "bbox": [float(x), float(y), float(w), float(h)]}
    return Face

def read_image_list(dir):
    image_list=[]
    img_path_list=os.listdir(dir)
    for img_name in img_path_list:
        img=cv2.imread(dir+"/"+img_name )#,cv2.IMREAD_GRAYSCALE
        gray_img= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image_list.append(Image(img,gray_img,img_name)) 
    return image_list
 


def parse_args():
    parser = argparse.ArgumentParser(description=' ')

 
    parser.add_argument('img_dir', type=str,nargs='?',default="data/Validation folder/")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print(args)

    output_path="results.json"
    input_dir=os.path.join(args.img_dir,'images')
    face_detect(input_dir , output_path,True,False)
    if NEED_VALIDATE:
        print(compute_f1(  output_path ))