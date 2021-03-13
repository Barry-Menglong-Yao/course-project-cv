

import json
import numpy as np
import os
from ccl.position import Position
from . import data_print

class OcrEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj,Position):
            return obj.to_result()
        return json.JSONEncoder.default(self, obj)



def save_block_list(block_list,dir):
    block_json_list=[]
    for position in  block_list :
        if position.label!=0:
            a={"bbox":position.to_result(),"name":position.character}
            block_json_list.append(a)
    with open(os.path.join(dir, 'results.json'), "w") as file:
        json.dump(block_json_list, file, cls=OcrEncoder)
