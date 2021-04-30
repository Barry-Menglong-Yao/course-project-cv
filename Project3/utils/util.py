import json
import numpy as np

class Face: 
    def __init__(self, img_idx,img_name,x,y,w,h):
        self.img_idx=img_idx
        self.img_name=img_name 
        self.x=x
        self.y=y 
        self.w=w
        self.h=h 

    def to_json(self):
        return {"iname": self.img_name, "bbox": [float(self.x), float(self.y), float(self.w), float(self.h)]}


class Encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, Face):
            return obj.to_json()
        return json.JSONEncoder.default(self, obj)