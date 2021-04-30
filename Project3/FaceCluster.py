from  FaceDetector import  *
import face_recognition

def face_cluster(img_dir):
    croped_face_list=crop_face(img_dir)
    k=extract_k(img_dir)
    face_recognition.face_encodings(img, boxes)

def extract_k(img_dir):
    dir_name=os.path.basename(img_dir )
    k=dir_name.split("_")[-1]
    return int(k)

def crop_face(img_dir):
    croped_face_list=[]
    face_list,image_list=face_detect(img_dir, None, False)
    for face in face_list:
        its_image=image_list[face.img_idx].pixels
        croped_face=im_crop(its_image,face.x,face.y,face.w,face.h)
        croped_face_list.append(croped_face)
    return croped_face_list

def im_crop(img,x1,y1,w,h):
    x2=x1+w 
    y2=y1+h
    return img[y1:y2, x1:x2, :]

if __name__ == "__main__":
    face_cluster("data/faceCluster_5"  )