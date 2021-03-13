

class UnionFindForest:
    def __init__(self):
        self.label_dic=dict()
        self.label=1

    def new_tree(self):
        self.label_dic[self.label]=self.label
  
        self.label+=1
        cur_label=self.label-1
        return cur_label

    def find_root(self,label):
        root=label 
        while self.label_dic[root]<root:
            root=self.label_dic[root]
        # print("{0} 's root is {1}  ".format(label,root))
        return root


    def set_root(self,root,leaf):
        while self.label_dic[leaf]<leaf:
            v=self.label_dic[leaf]
            self.label_dic[leaf]=root 
            leaf=v 
        self.label_dic[leaf]=root

    def union_tree(self,m,n): 
        root=self.find_root(m)
        if(root!=n):
            s=self.find_root(n)
            if(root>s):
                root=s
            self.set_root(root,n)
        self.set_root(root,m)
        return root
            

    def flatten(self):
        for i in range(1,self.label):
            self.label_dic[i]=self.label_dic[self.label_dic[i]]
        

    

