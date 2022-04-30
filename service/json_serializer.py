import json
import numpy as np


class RespondPkg:
    def __init__(self, pkg_type, result):
        self.pkg_type = pkg_type
        self.final_boxes = result[0]
        self.final_class_ids = result[1]


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def respond2json(obj: RespondPkg):
    return {
        'pkg_type': obj.pkg_type,
        'final_boxes': obj.final_boxes,
        "final_class_ids": obj.final_class_ids
    }


def pack_json(json_data):
    str_json = json.dumps(json_data, cls=NpEncoder)
    byte_json = str.encode(str_json)
    length = str(len(byte_json)).ljust(8)
    byte_length = str.encode(length)
    return byte_length, byte_json


def unpack_json(bytes_json):
    str_json = bytes.decode(bytes_json)
    pck = json.loads(str_json)
    return pck


