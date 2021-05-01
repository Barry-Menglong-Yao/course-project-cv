from  FaceDetector import  *
import face_recognition
from cluster.k_means import * 
import argparse

def face_cluster(img_dir):
    face_list,image_list=face_detect(img_dir, None, False,False)
    k=extract_k(img_dir)
    feature_list=gen_face_features(face_list,image_list)
    cluster_list=k_means(feature_list,k)

def gen_face_features(face_list,image_list):
    feature_list=[]
    for i in range(len(face_list)):
        face=face_list[i]
        boxes=gen_boxes(face)
        feature=face_recognition.face_encodings(image_list[face.img_idx].rgb_img, boxes)
        feature_list.append(feature[0])
    return feature_list


def gen_boxes(face):
    boxes=[]
    boxes.append(face.to_box())
    return boxes

def extract_k(img_dir):
    dir_name=os.path.basename(img_dir )
    k=dir_name.split("_")[-1]
    return int(k)

def crop_face(img_dir):
    croped_face_list=[]
    face_list,image_list=face_detect(img_dir, None, False,False)
    for face in face_list:
        its_image=image_list[face.img_idx].pixels
        croped_face=im_crop(its_image,face.x,face.y,face.w,face.h)
        croped_face_list.append(croped_face)
    return croped_face_list,face_list

def im_crop(img,x1,y1,w,h):
    x2=x1+w 
    y2=y1+h
    return img[y1:y2, x1:x2, :]



def parse_args():
    parser = argparse.ArgumentParser(description=' ')

 
    parser.add_argument('--img_dir', type=str,default="data/faceCluster_5")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print(args)
    face_cluster(args.img_dir  )