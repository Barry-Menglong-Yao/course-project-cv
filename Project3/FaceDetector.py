import cv2
import os
import json 
from utils.util import Face
from utils.util import Encoder

class Image:
    def __init__(self, img,name):
        self.pixels=img
        self.name=name 


def face_detect(img_dir,output_path,is_save_json):
    face_list=[]
    image_list=read_image_list(img_dir)
    face_cascade=cv2.CascadeClassifier("data/haarcascade_frontalface_default.xml")
    for idx in range(len(image_list)):
        image=image_list[idx]
        face_position_list=face_cascade.detectMultiScale(image.pixels) #,minSize=(5, 5)
        for face_position in face_position_list:
            x,y,w,h=face_position 
            face=Face(idx,image.name,x,y,w,h)
            face_list.append(face)
    if is_save_json:
        with open(output_path, 'w') as f:
            json.dump(face_list, f, cls=Encoder)
    return face_list,image_list
 
    

def gen_Face(idx,face,img_name):
    x,y,w,h=face 
    Face(idx,img_name,x,y,w,h)
    # json_object = {"iname": img_name, "bbox": [float(x), float(y), float(w), float(h)]}
    return Face

def read_image_list(dir):
    image_list=[]
    img_path_list=os.listdir(dir)
    for img_name in img_path_list:
        img=cv2.imread(dir+"/"+img_name,cv2.IMREAD_GRAYSCALE )
        image_list.append(Image(img,img_name)) 
    return image_list



if __name__ == "__main__":
    face_detect("data/Validation folder/images","output/detect/results.json" ,True)