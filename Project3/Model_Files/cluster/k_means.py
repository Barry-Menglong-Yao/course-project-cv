import random
import numpy 
import copy 
import unittest
class Cluster:
    def __init__(self, cluster_no,centroid ):
        self.cluster_no=cluster_no
        self.centroid=centroid
        
        self.element_id_list=[]

    def add_element(self,element_id):
        self.element_id_list.append(element_id)

    def empty(self):
        self.element_id_list=[]

def k_means(sample_list,k):
    cluster_list=init_cluster_list(sample_list,k)
 
    epoch=0
    while True:
        new_cluster_list=cluster_sample(cluster_list,sample_list )
        if equal(new_cluster_list,cluster_list) :
            print(f'stop at epoch {epoch}, label_list {cluster_list}')
            break
        else:
            cluster_list=new_cluster_list
            cluster_list=update_cluster_centroid(cluster_list,sample_list  )
            epoch+=1
    return  cluster_list


def equal(new_cluster_list,old_cluster_list):
    is_equal=True
    for old_cluster in old_cluster_list:
        has_found_same_one=False
        for new_cluster in new_cluster_list:
            if numpy.array_equal(old_cluster.centroid,new_cluster.centroid) and   numpy.array_equal( old_cluster.element_id_list,new_cluster.element_id_list) :
                has_found_same_one=True
                break
        if has_found_same_one==False:
            is_equal=False 
            break
    return is_equal 

def update_cluster_centroid(cluster_list ,sample_list):
    for cluster in cluster_list:
        element_list=gen_element_list(cluster.element_id_list,sample_list)
        cluster.centroid=numpy.average(element_list,axis=0)
    return cluster_list

def gen_element_list(element_id_list,sample_list):
    element_list=[]
    for element_id in element_id_list:
        element_list.append(sample_list[element_id])
    return element_list

def init_cluster_list(sample_list,k):
    idx_list=random.sample(range(len(sample_list)), k)
    cluster_list=[]
    for i,random_idx in enumerate(idx_list):
 
        cluster=Cluster(i,sample_list[random_idx])
        cluster.add_element( random_idx )
        cluster_list.append(cluster)
    return cluster_list

def empty_cluster_list(cluster_list):
    for cluster in cluster_list:
        cluster.empty()

def cluster_sample(cluster_list,sample_list ):
    new_cluster_list=copy.deepcopy(cluster_list)
    empty_cluster_list(new_cluster_list)
    for i,sample in enumerate(sample_list):
        nearest_cluster_id=find_nearest_cluster(sample,new_cluster_list)
        new_cluster_list[nearest_cluster_id].add_element(i)
    return new_cluster_list

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
 






class TestStringMethods(unittest.TestCase):
 
    def test_1(self):
        sample_list=[1,5,101,105]
        cluster_list=k_means(sample_list,2)
        self.assertTrue(cluster_list[0].centroid==3 or cluster_list[1].centroid==3)
        self.assertTrue(cluster_list[0].centroid==103 or cluster_list[1].centroid==103)
 
    def test_2(self):
        data =numpy.array([[1,5],[101,105],[109,111],[7,15]])
         
        cluster_list=k_means(data,2)
        self.assertTrue(numpy.array_equal(cluster_list[0].centroid,[105,108]) or numpy.array_equal(cluster_list[1].centroid,[105,108]))
        self.assertTrue(numpy.array_equal(cluster_list[0].centroid,[4,10]) or numpy.array_equal(cluster_list[1].centroid,[4,10]))


    def test_3(self):
        x = numpy.random.randint(25,100,25)
        y = numpy.random.randint(175,255,25)
        z = numpy.hstack((x,y))
        z = z.reshape((50,1))
        z = numpy.float32(z)
        cluster_list=k_means(z,2)
        self.assertTrue(cluster_list[0].centroid<100 or cluster_list[1].centroid<100 )


    def test_4(self):
        X = numpy.random.randint(25,50,(25,2))
        Y = numpy.random.randint(60,85,(25,2))
        Z = numpy.vstack((X,Y))

        # convert to np.float32
        Z = numpy.float32(Z)
        cluster_list=k_means(Z,2)
        self.assertTrue(cluster_list[0].centroid[0]<100 or cluster_list[1].centroid[0]<100 )


if __name__ == '__main__':
    unittest.main()
