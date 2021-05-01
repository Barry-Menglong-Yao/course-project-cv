import random
import numpy 


class Cluster:
    def __init__(self, cluster_no,centroid ):
        self.cluster_no=cluster_no
        self.centroid=centroid
        self.element_list=[]

    def add_element(self,element):
        self.element_list.append(element)

    def empty(self):
        self.element_list=[]

def k_means(sample_list,k):
    cluster_list=init_cluster_list(sample_list,k)
    label_list=[]
    epoch=0
    while:
        new_label_list=label_sample(cluster_list,sample_list )
        if numpy.equal(new_label_list,label_list):
            print(f'stop at epoch {epoch}, label_list {label_list}')
            break
        else:
            label_list=new_label_list
            cluster_list=update_cluster_list(label_list,sample_list )
            epoch+=1
    return label_list,cluster_list


def update_cluster_list(label_list,sample_list ):
    for sample in sample_list:

def init_cluster_list(sample_list,k):
    cluster_list=[]
    for i in range(k):
        random_idx=random.randint(0,len(sample_list)-1)
        cluster=Cluster(i,sample_list[random_idx])
        cluster_list.append(cluster)
    return cluster_list

def empty_cluster_list(cluster_list):
    for cluster in cluster_list:
        cluster.empty()

def label_sample(cluster_list,sample_list ):
    empty_cluster_list(cluster_list)
    for sample in sample_list:
        nearest_cluster_id=find_nearest_cluster(sample,cluster_list)
        cluster_list[nearest_cluster_id].add_element(sample)
        label_list.append(nearest_cluster_id)
    return cluster_list

def find_nearest_cluster(sample,cluster_list):
    smallest_distance=1000000000
    nearest_cluster_id=-1
    for i in range(len(cluster_list)):
        cluster=cluster_list[i]
        distance= numpy.linalg.norm( sample-cluster.centroid)
        if distance<smallest_distance:
            smallest_distance=distance  
            nearest_cluster_id=i
    return nearest_cluster_id
 