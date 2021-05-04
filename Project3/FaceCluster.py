from  FaceDetector import  *
import face_recognition
from Model_Files.cluster.k_means import * 
import argparse
import json 
from Model_Files.utils.concat_util import * 

def face_cluster(img_dir):
    face_list,image_list=face_detect(img_dir, None, False,False)
    k=extract_k(img_dir)
    feature_list=gen_face_features(face_list,image_list)
    cluster_list=k_means(feature_list,k)
    gen_clusters_output(cluster_list,face_list,image_list)

def gen_clusters_output(cluster_list,face_list,image_list):
    gen_clusters_json(cluster_list,face_list)
    is_save_clustered_imgs=False
    if is_save_clustered_imgs:
        concat_multi_faces_by_cluster(cluster_list,face_list,image_list)


def gen_clusters_json(cluster_list,face_list):
    json_array=[]
    for cluster in cluster_list:
        json_object=gen_cluster_json(cluster,face_list)
        json_array.append(json_object)
    output_json = "clusters.json"
    with open(output_json, 'w') as f:
        json.dump(json_array, f)

def gen_cluster_json(cluster,face_list):
    element_name_list=[]
    for element_id in cluster.element_id_list:
        element_name_list.append(face_list[element_id].img_name)
    return {"cluster_no":cluster.cluster_no,"elements":element_name_list}

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




def parse_args():
    parser = argparse.ArgumentParser(description=' ')

 
    parser.add_argument('img_dir', type=str,nargs='?',default="data/faceCluster_5")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print(args)
    face_cluster(args.img_dir  )